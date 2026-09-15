"""Prepares a resolved file for attaching to an outbound task."""

from __future__ import annotations

import os

from assistant.files.file_policy import FilePolicy


class AttachmentManager:
    def __init__(self, policy: FilePolicy):
        self.policy = policy

    def prepare(self, path: str) -> dict:
        if not self.policy.is_allowed(path):
            raise PermissionError(f"File not in an allowed location: {path}")
        if not self.policy.check_size(path):
            raise ValueError(f"File exceeds size limit: {path}")
        return {"path": path, "filename": os.path.basename(path), "size": os.path.getsize(path)}
