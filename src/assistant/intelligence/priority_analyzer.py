"""
Priority Analyzer & Background Batch Processor
===============================================
Periodically batches pending messages across all sources (Gmail, Telegram,
WhatsApp, SMS), extracts structured semantic intelligence via the Message
Intelligence LLM, applies user preferences and rules with strict precedence,
and updates the Priority Inbox without blocking the application.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from assistant.intelligence.message_intelligence_models import (
    BatchAnalysisResponse,
    MessageAnalysisItem,
)
from assistant.intelligence.priority_inbox import PriorityInbox
from assistant.intelligence.priority_rules import PriorityRulesManager
from assistant.llm.llm_client import (
    LLMAuthenticationError,
    LLMClient,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    classify_llm_error,
)
from assistant.memory.user_profile import UserProfile

logger = logging.getLogger("assistant.priority_analyzer")


_SYSTEM_PROMPT = """\
You are an expert Message Intelligence Analyzer for a personal AI assistant.
Your job is to analyze incoming messages across channels (Gmail, Telegram, WhatsApp, SMS) and extract structured semantic intelligence.

Output must be a JSON object with an "items" array, where each object has:
- message_id: string matching the input item id
- category: one of [WORK, PERSONAL, PROMOTIONAL, TRANSACTIONAL, SOCIAL, NEWSLETTER, OTHER]
- intent: one of [INTERVIEW, MEETING, TASK, NOTIFICATION, INQUIRY, CHAT, SPAM, OTHER]
- importance: one of [HIGH, MEDIUM, LOW]
- urgency: one of [HIGH, MEDIUM, LOW]
- requires_action: boolean
- spam: boolean
- scam: boolean (true if phishing, fraudulent, or suspicious risk)
- risk_score: float between 0.0 and 1.0
- deadline: explicit ISO-8601 string or date/time (e.g. '2026-09-15T09:00:00' or '2026-09-15') if mentioned directly or relative to the message timestamp and reference time. If relative words like 'tomorrow at 9am' are used, resolve to actual date/time based on the message timestamp. STRICT: NEVER invent a time if only a date or vague relative time is given.
- reason: concise explanation for classification (max 15 words)

STRICT INSTRUCTIONS:
1. Return ONLY valid JSON matching the schema. No conversational prose or markdown wrappers outside the JSON.
2. Evaluate spam and scam independently from importance. A scam alert can be HIGH importance/urgency to notify the user, while scam=true.
3. For meetings, appointments, tasks, or interviews with a date/time (e.g. tomorrow morning at 9am), intent must be MEETING or INTERVIEW or TASK, requires_action must be true, and urgency/importance should reflect the upcoming commitment.
"""



class PriorityAnalyzer:
    def __init__(
        self,
        priority_inbox: PriorityInbox,
        rules_manager: PriorityRulesManager | None = None,
        user_profile: UserProfile | None = None,
        llm_client: LLMClient | None = None,
        batch_size: int = 50,
        max_batches_per_cycle: int = 3,
        max_retries: int = 1,
        timeout_seconds: float = 30.0,
        stale_timeout_seconds: float = 900.0,
    ):
        self.priority_inbox = priority_inbox
        self.rules_manager = rules_manager or PriorityRulesManager()
        self.user_profile = user_profile
        self.llm_client = llm_client
        self.batch_size = max(1, batch_size)
        self.max_batches_per_cycle = max(1, max_batches_per_cycle)
        self.max_retries = max(0, max_retries)
        self.timeout_seconds = timeout_seconds
        self.stale_timeout_seconds = stale_timeout_seconds

    def recover_stale_processing(self, timeout_seconds: float | None = None) -> int:
        timeout = timeout_seconds if timeout_seconds is not None else self.stale_timeout_seconds
        stale_before = time.time() - timeout
        recovered = self.priority_inbox.recover_stale_processing(stale_before)
        if recovered > 0:
            logger.info("Priority analyzer: recovered %d stale processing messages to PENDING", recovered)
        return recovered

    def process_cycle(self) -> dict[str, Any]:
        """Execute one analysis cycle.

        Key invariant:
        If pending_count == 0, returns immediately with ZERO LLM calls and zero retries.
        """
        pending_count = self.priority_inbox.count_pending_messages()
        if pending_count == 0:
            logger.debug("Priority analyzer: 0 pending messages; skipping LLM call")
            return {"status": "idle", "pending": 0, "processed": 0}

        logger.info("Priority analyzer: found %d pending messages", pending_count)
        total_processed = 0
        total_completed = 0
        total_deferred = 0

        # Process up to max_batches_per_cycle to prevent unbounded LLM calls in one cycle
        for batch_num in range(self.max_batches_per_cycle):
            candidates = self.priority_inbox.get_pending_messages(limit=self.batch_size)
            if not candidates:
                break

            candidate_ids = [c["id"] for c in candidates]
            locked_ids = self.priority_inbox.mark_messages_processing(candidate_ids)
            if not locked_ids:
                break

            batch = [c for c in candidates if c["id"] in locked_ids]
            logger.info("Priority analyzer: processing batch %d (size=%d)", batch_num + 1, len(batch))

            batch_res = self._process_single_batch(batch)
            total_processed += len(batch)
            total_completed += batch_res.get("completed", 0)
            total_deferred += batch_res.get("deferred", 0)

            # If fewer than batch_size were fetched, all pending have been handled
            if len(batch) < self.batch_size:
                break

        return {
            "status": "completed",
            "pending_initial": pending_count,
            "processed": total_processed,
            "completed": total_completed,
            "deferred": total_deferred,
        }

    def _build_prompt_messages(self, batch: list[dict[str, Any]]) -> list[Any]:
        # Extract minimal relevant preferences without dumping entire USER.md
        important_contacts: list[str] = []
        if self.user_profile:
            try:
                important_contacts = self.user_profile.important_contacts()
            except Exception:
                pass

        active_rules = self.rules_manager.list_rules(active_only=True)
        rules_summary = [
            f"- Rule #{r['id']} ({r['rule_type']}): '{r['pattern']}' -> {r['target_level']}"
            for r in active_rules
        ]

        context_lines = []
        if important_contacts:
            context_lines.append(f"User Important Contacts: {', '.join(important_contacts)}")
        if rules_summary:
            context_lines.append("Active User Priority Rules:\n" + "\n".join(rules_summary))

        ref_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        context_lines.append(f"Current Reference Time: {ref_now}")

        context_block = ("\n\nContext:\n" + "\n".join(context_lines)) if context_lines else ""

        # Build sanitized batch payload for LLM with timestamp for relative date resolution
        payload = [
            {
                "message_id": str(msg["id"]),
                "source": msg.get("source", ""),
                "sender": msg.get("sender", ""),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg.get("ts", time.time()))),
                "content": msg.get("content", ""),
            }
            for msg in batch
        ]

        user_prompt = f"Analyze the following {len(payload)} incoming messages:{context_block}\n\nMessages:\n{json.dumps(payload, ensure_ascii=False)}"
        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

    def _call_llm_with_resilience(self, prompt_messages: list[Any]) -> str:
        if self.llm_client is None or not getattr(self.llm_client, "online", True):
            raise LLMAuthenticationError("PRIORITY_GROQ_API_KEY is not configured or offline")

        def invoke():
            model = getattr(self.llm_client, "model", None)
            if model is not None and hasattr(model, "invoke"):
                res = model.invoke(prompt_messages)
                return getattr(res, "content", str(res))
            # Fallback to generate
            prompt_str = "\n".join(getattr(m, "content", str(m)) for m in prompt_messages)
            return self.llm_client.generate(prompt_str)

        attempt = 0

        while True:
            attempt += 1
            try:
                return invoke()
            except Exception as exc:
                classified = classify_llm_error(exc) if not isinstance(exc, LLMError) else exc
                if isinstance(classified, LLMNetworkError):
                    if attempt <= self.max_retries:
                        logger.warning("Priority analyzer: temporary network failure (attempt %d); retrying in 1s", attempt)
                        time.sleep(1.0)
                        continue
                    logger.warning("Priority analyzer: network failure after %d attempts; deferring batch", attempt)
                    raise classified

                if isinstance(classified, LLMRateLimitError):
                    retry_after = getattr(classified, "retry_after", None)
                    if retry_after is not None and 0 < retry_after <= 5.0 and attempt <= self.max_retries:
                        logger.warning("Priority analyzer: rate limited; retrying after %.1fs", retry_after)
                        time.sleep(retry_after)
                        continue
                    logger.warning("Priority analyzer: rate limited with delay %s; deferring batch", retry_after)
                    raise classified

                if isinstance(classified, LLMAuthenticationError):
                    logger.error("Priority analyzer: authentication failure; deferring batch without loop")
                    raise classified

                # General service or other error
                if attempt <= self.max_retries:
                    time.sleep(1.0)
                    continue
                raise classified

    def _parse_llm_response(self, raw_text: str) -> dict[str, MessageAnalysisItem]:
        """Extract and validate MessageAnalysisItem per message.

        Never partially accepts an individual message. A message is valid ONLY
        when its entire structured schema validates.
        """
        cleaned = raw_text.strip()
        # Strip markdown code fencing if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
        items_raw: list[Any] = []
        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            items_raw = data["items"]
        elif isinstance(data, list):
            items_raw = data
        elif isinstance(data, dict):
            items_raw = [data]

        results: dict[str, MessageAnalysisItem] = {}
        for item in items_raw:
            if not isinstance(item, dict):
                continue
            try:
                parsed = MessageAnalysisItem.model_validate(item)
                results[str(parsed.message_id)] = parsed
            except Exception as val_err:
                msg_id = str(item.get("message_id", "unknown"))
                logger.warning("Priority analyzer: message id=%s failed schema validation: %s", msg_id, val_err)

        return results

    def _process_single_batch(self, batch: list[dict[str, Any]]) -> dict[str, int]:
        batch_ids = [m["id"] for m in batch]
        prompt_messages = self._build_prompt_messages(batch)

        try:
            raw_text = self._call_llm_with_resilience(prompt_messages)
        except (LLMNetworkError, LLMRateLimitError, LLMAuthenticationError, LLMError) as exc:
            self.priority_inbox.defer_processing_messages(batch_ids, error=f"llm_{type(exc).__name__}")
            return {"completed": 0, "deferred": len(batch_ids)}
        except Exception as exc:
            self.priority_inbox.defer_processing_messages(batch_ids, error=f"error_{type(exc).__name__}")
            return {"completed": 0, "deferred": len(batch_ids)}

        # Parse structured output
        parsed_items: dict[str, MessageAnalysisItem] = {}
        try:
            parsed_items = self._parse_llm_response(raw_text)
        except Exception as exc:
            logger.warning("Priority analyzer: malformed LLM response; attempting single repair retry")
            # Try 1 repair retry
            try:
                repair_prompt = [
                    SystemMessage(content="You must format the following output as valid JSON matching schema {\"items\": [...]}. Return ONLY raw JSON."),
                    HumanMessage(content=f"Fix this JSON:\n{raw_text}"),
                ]
                repaired_text = self._call_llm_with_resilience(repair_prompt)
                parsed_items = self._parse_llm_response(repaired_text)
            except Exception:
                logger.error("Priority analyzer: repair failed; deferring entire batch")
                self.priority_inbox.defer_processing_messages(batch_ids, error="malformed_json")
                return {"completed": 0, "deferred": len(batch_ids)}

        # Load active rules for precedence evaluation
        active_rules = self.rules_manager.list_rules(active_only=True)
        completed_count = 0
        deferred_count = 0

        # Per-message evaluation & commit
        for msg in batch:
            msg_id_str = str(msg["id"])
            analysis = parsed_items.get(msg_id_str)

            # Strict validation rule: A message is COMPLETED ONLY when its entire schema validated
            if analysis is None:
                self.priority_inbox.defer_processing_messages([msg["id"]], error="invalid_or_missing_analysis")
                deferred_count += 1
                continue

            final_level, matched_rule_id, override_reason = self.compute_final_priority(
                msg=msg,
                analysis=analysis,
                active_rules=active_rules,
            )

            model_name = ""
            if self.llm_client:
                m_attr = getattr(self.llm_client, "model_name", None)
                if isinstance(m_attr, str):
                    model_name = m_attr
                elif isinstance(getattr(self.llm_client, "model", None), str):
                    model_name = self.llm_client.model
                else:
                    model_name = "analyzer-model"

            self.priority_inbox.update_analysis_success(
                id=msg["id"],
                analysis=analysis.model_dump(),
                final_level=final_level,
                user_rule_id=matched_rule_id,
                override_reason=override_reason,
                model=model_name,
                version="1.0",
            )
            completed_count += 1

        logger.info("Priority analyzer: batch finished size=%d completed=%d deferred=%d", len(batch), completed_count, deferred_count)
        return {"completed": completed_count, "deferred": deferred_count}

    def compute_final_priority(
        self,
        msg: dict[str, Any],
        analysis: MessageAnalysisItem,
        active_rules: list[dict[str, Any]] | None = None,
    ) -> tuple[str, int | None, str]:
        """Strict 6-tier precedence hierarchy:

        1. Explicit message-level user override
        2. Explicit persistent user rule (priority_rules)
        3. Stable user preference (UserProfile important contacts)
        4. Message Intelligence analysis result
        5. Existing deterministic score
        6. Default (LOW)
        """
        # Tier 1: Explicit message-level override
        override = msg.get("user_override_level")
        if override and override.upper() in ("HIGH", "MEDIUM", "LOW", "IGNORE"):
            return override.upper(), None, "Explicit message-level user override"

        # Tier 2: Explicit persistent user rule (deterministic conflict resolution)
        matched_rule = self.rules_manager.match_rule(
            sender=msg.get("sender", ""),
            content=msg.get("content", ""),
            category=analysis.category,
            rules=active_rules,
        )
        if matched_rule:
            target = matched_rule.get("target_level", "HIGH").upper()
            rule_id = matched_rule.get("id")
            reason = f"Matched user rule #{rule_id} ({matched_rule.get('pattern')} -> {target})"
            return target, rule_id, reason

        # Tier 3: Stable user preferences (important contacts)
        if self.user_profile:
            sender_lower = (msg.get("sender") or "").lower()
            try:
                important = [c.lower() for c in self.user_profile.important_contacts()]
                if any(ic in sender_lower or sender_lower in ic for ic in important if ic):
                    return "HIGH", None, f"Sender matches stable user preference important contacts"
            except Exception:
                pass

        # Tier 4: Message Intelligence semantic classification
        if analysis.importance == "HIGH" or analysis.urgency == "HIGH":
            return "HIGH", None, f"Semantic intelligence: {analysis.reason}"
        if analysis.importance == "MEDIUM" or analysis.urgency == "MEDIUM":
            return "MEDIUM", None, f"Semantic intelligence: {analysis.reason}"
        if analysis.importance == "LOW" and analysis.urgency == "LOW":
            return "LOW", None, f"Semantic intelligence: {analysis.reason}"

        # Tier 5: Existing deterministic score/level
        det_level = msg.get("level")
        if det_level and det_level.upper() in ("HIGH", "MEDIUM", "LOW"):
            return det_level.upper(), None, "Deterministic priority fallback"

        # Tier 6: Default
        return "LOW", None, "Default priority level"


class PriorityAnalyzerWorker:
    def __init__(
        self,
        analyzer: PriorityAnalyzer,
        interval_minutes: float = 30.0,
        run_on_startup: bool = False,
    ):
        self.analyzer = analyzer
        self.interval_seconds = max(1.0, interval_minutes * 60.0)
        self.run_on_startup = run_on_startup
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="priority-analyzer-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("Priority analyzer worker started (interval=%.1f min, startup=%s)", self.interval_seconds / 60.0, self.run_on_startup)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # 1. Recover stale processing messages from previous crash
        try:
            self.analyzer.recover_stale_processing()
        except Exception as exc:
            logger.error("Priority analyzer: error during startup stale recovery: %s", exc)

        # 2. Run initial cycle only if explicitly configured
        if self.run_on_startup:
            try:
                self.analyzer.process_cycle()
            except Exception as exc:
                logger.error("Priority analyzer: error during startup cycle: %s", exc)

        # 3. Main wait loop using stoppable threading.Event
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval_seconds):
                break
            try:
                self.analyzer.process_cycle()
            except Exception as exc:
                logger.error("Priority analyzer: cycle error: %s", exc)
