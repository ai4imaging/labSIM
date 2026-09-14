from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from storage.trajectories import TRAJECTORY_FILENAME


@dataclass(slots=True)
class TraceWriter:
    """Persist system prompt and conversation events for a run."""

    trace_dir: Path
    trajectory_filename: str = TRAJECTORY_FILENAME
    conversation_path: Path = field(init=False)
    _conversation_file: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_path = self.trace_dir / self.trajectory_filename
        self._conversation_file = self.conversation_path.open("w", encoding="utf-8")

    def write_message(self, message: dict[str, Any]) -> None:
        record = {"message": message}
        self.write_event("message", record)

    def write_llm_request(
        self,
        *,
        turn: int,
        provider: str,
        model_id: str | None,
        message_count: int,
        tool_names: list[str],
        system_prompt_chars: int,
    ) -> None:
        """Mark the point at which a turn's request went out.

        bioSIM seam. The trace already holds every message, but nothing in it says where
        one turn ends and the next begins, which model answered, or what tools were on
        offer at the time. Without that, a transcript can be read but a run cannot be
        reconstructed — and "why did it not call the grounding tool" is unanswerable if
        there is no record of whether the tool was even advertised.

        The messages themselves are not repeated here; they are already in the file as
        `message` events, and duplicating them would multiply the trace by the number of
        turns.
        """
        self.write_event(
            "llm_request",
            {
                "turn": turn,
                "provider": provider,
                "model_id": model_id,
                "message_count": message_count,
                "tool_names": tool_names,
                "system_prompt_chars": system_prompt_chars,
            },
        )

    def write_llm_result(
        self,
        *,
        turn: int,
        duration_s: float,
        text_chars: int,
        tool_calls: list[str],
        usage: dict[str, Any] | None,
    ) -> None:
        """What came back, and how long it took. Pairs with `write_llm_request`."""
        self.write_event(
            "llm_result",
            {
                "turn": turn,
                "duration_s": round(duration_s, 3),
                "text_chars": text_chars,
                "tool_calls": tool_calls,
                "usage": usage or {},
            },
        )

    def write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": time.time(),
            "type": event_type,
            **payload,
        }
        self._conversation_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._conversation_file.flush()

    def close(self) -> None:
        with suppress(Exception):
            self._conversation_file.close()
