"""File Security — allowed roots + size limits (Part 3, Section 19)."""

from __future__ import annotations

import os


class FilePolicy:
    def __init__(self, allowed_roots: list[str], max_file_size_mb: int = 25):
        self.allowed_roots = [os.path.abspath(os.path.expanduser(r)) for r in allowed_roots]
        self.max_bytes = max_file_size_mb * 1024 * 1024

    def is_allowed(self, path: str) -> bool:
        abs_path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
        return any(os.path.commonpath([abs_path, root]) == root for root in self.allowed_roots)

    def check_size(self, path: str) -> bool:
        try:
            return os.path.getsize(path) <= self.max_bytes
        except OSError:
            return False
