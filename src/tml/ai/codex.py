from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .client import AiRequest, AiResponse


class CodexAiClient:
    """Small Codex CLI adapter; future app-server clients can implement AiClient."""

    def call(self, request: AiRequest) -> AiResponse:
        with tempfile.TemporaryDirectory(prefix="tml-codex-") as tmp:
            response_path = Path(tmp) / "response.txt"
            result = subprocess.run(
                [
                    "codex",
                    "--ask-for-approval",
                    "never",
                    "exec",
                    "--sandbox",
                    "read-only",
                    "--model",
                    request.model,
                    "--output-last-message",
                    str(response_path),
                    "-",
                ],
                input=request.prompt,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"Codex backend failed: {message}")
            return AiResponse(
                text=response_path.read_text(encoding="utf-8"),
                metadata={"backend": "codex", "model": request.model},
            )
