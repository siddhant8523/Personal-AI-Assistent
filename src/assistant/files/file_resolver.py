"""
File Resolver (Part 3, Section 17-18)
=========================================
Finds a candidate file matching a fuzzy user description within permitted
locations. Ambiguity -> ask, never guess.
"""

from __future__ import annotations

import os

from assistant.files.file_policy import FilePolicy


class AmbiguousFileError(Exception):
    def __init__(self, candidates: list[str]):
        self.candidates = candidates
        super().__init__(f"Ambiguous file reference, candidates: {candidates}")


class FileResolver:
    def __init__(self, policy: FilePolicy):
        self.policy = policy

    def find(self, query: str) -> str | None:
        query_lower = query.lower()
        candidates = []
        for root in self.policy.allowed_roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _dirs, filenames in os.walk(root):
                for fname in filenames:
                    if query_lower in fname.lower():
                        candidates.append(os.path.join(dirpath, fname))

        if not candidates:
            return None
        if len(candidates) > 1:
            raise AmbiguousFileError(candidates)
        return candidates[0]
