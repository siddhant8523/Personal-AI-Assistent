"""
Semantic Memory (Part 2, Section 19)
=======================================
A real deployment would swap this for an embedding/vector store. To keep
this project runnable without extra heavy ML dependencies or an embedding
API key, retrieval here uses simple token-overlap (Jaccard) scoring over
stored text entries. The interface (add/search) is what matters — the
Context Builder doesn't know or care which implementation sits behind it.
"""

from __future__ import annotations

import time

from assistant.storage.db import get_connection


def _tokenize(text: str) -> set[str]:
    return set(w.strip(".,!?\"'").lower() for w in text.split() if len(w) > 2)


class SemanticMemory:
    def add(self, text: str, tags: str = "") -> None:
        conn = get_connection()
        conn.execute("INSERT INTO semantic_memory (text, tags, ts) VALUES (?,?,?)", (text, tags, time.time()))
        conn.commit()

    def search(self, query: str, top_k: int = 5) -> list[str]:
        conn = get_connection()
        rows = conn.execute("SELECT text FROM semantic_memory").fetchall()
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored = []
        for r in rows:
            t_tokens = _tokenize(r["text"])
            if not t_tokens:
                continue
            overlap = len(q_tokens & t_tokens) / len(q_tokens | t_tokens)
            if overlap > 0:
                scored.append((overlap, r["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]
