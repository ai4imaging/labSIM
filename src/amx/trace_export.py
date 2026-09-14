"""Turning a run's trace into something a person can actually read.

Articraft writes one JSON object per line to `trajectory.jsonl`. That is the right format
to write and the wrong one to read: a forty-turn run is a few megabytes on one line each,
the tool results are JSON strings nested inside JSON, and the code the agent wrote is
buried inside the arguments of a `replace` call. Answering "what did it do on turn twelve
and why" means writing a script, every time.

This expands the same events into a directory per turn:

    turns/turn-012/
      request.json        which model, how many messages, which tools were on offer
      response.json       text, tool calls, token usage, wall time
      thinking.txt        reasoning, when the provider returns any
      tool-calls.json     the calls as issued, arguments parsed
      tool-results/       one file per result, with the signal blocks left as text
      model.py            the file as it stood after this turn's edits
      injected.txt        the reminders the harness added and the model did not ask for

Nothing is computed here that is not already in the trace. This is a reformatting, so it
can be re-run on an old trace and cannot disagree with the source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRAJECTORY = "trajectory.jsonl"

INJECTED_MARKERS = (
    "<compile_required>",
    "<grounding_required>",
    "<final_response_required>",
    "<edit_code_guidance>",
    "<api_error_guidance>",
    "<code_contract_guidance>",
    "<grounding_renders>",
    "<no_action_recovery>",
)
"""Openers the harness uses for messages it inserts on the model's behalf.

Separating these out is what makes a transcript legible: a `user` message in this file is
either the task or the harness talking, and conflating the two makes it look as though
someone kept interrupting.
"""


@dataclass
class Turn:
    number: int
    request: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    assistant: dict[str, Any] | None = None
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    injected: list[str] = field(default_factory=list)
    user_messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        calls = ", ".join(call.get("name", "?") for call in self.tool_calls) or "(none)"
        duration = f"{self.result.get('duration_s', 0):.1f}s" if self.result else "?"
        usage = (self.result or {}).get("usage", {})
        tokens = usage.get("total_tokens") or usage.get("completion_tokens") or 0
        return f"turn {self.number:>3}  {duration:>7}  {tokens:>7} tok  tools: {calls}"


@dataclass
class Trace:
    turns: list[Turn]
    events: int
    model_id: str | None = None
    provider: str | None = None

    def to_text(self) -> str:
        head = [
            f"{len(self.turns)} turns from {self.events} events"
            + (f", {self.provider}/{self.model_id}" if self.model_id else ""),
            "",
        ]
        return "\n".join(head + [turn.summary() for turn in self.turns])


def _trace_text(path: Path) -> str:
    """The trace as text, whether or not Articraft compressed it.

    A run that finished has its trace promoted into the record tree and zstd-compressed;
    only a run that failed leaves a plain `trajectory.jsonl` behind. Reading just the plain
    one means every tool built on this sees failures and nothing else.
    """
    if path.suffix != ".zst":
        return path.read_text(errors="replace")
    import zstandard  # noqa: PLC0415

    with path.open("rb") as handle:
        with zstandard.ZstdDecompressor().stream_reader(handle) as reader:
            return reader.read().decode("utf-8", errors="replace")


def read_trace(trace_dir: Path) -> Trace:
    """Group a `trajectory.jsonl` into turns.

    Turn boundaries come from `llm_request` events. A trace written before those existed
    has none, and everything lands in a single turn zero — which is honest: the
    information to split it was never recorded.
    """
    path = Path(trace_dir)
    if path.is_dir():
        path = path / TRAJECTORY
        if not path.is_file() and path.with_suffix(f"{path.suffix}.zst").is_file():
            path = path.with_suffix(f"{path.suffix}.zst")
    if not path.is_file():
        raise FileNotFoundError(f"no trace at {path}")

    turns: list[Turn] = []
    current = Turn(number=0)
    events = 0
    model_id: str | None = None
    provider: str | None = None

    for line in _trace_text(path).splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        events += 1
        kind = record.get("type")

        if kind == "llm_request":
            if current.request is not None or current.tool_calls or current.user_messages:
                turns.append(current)
            current = Turn(number=int(record.get("turn", len(turns) + 1)), request=record)
            model_id = record.get("model_id") or model_id
            provider = record.get("provider") or provider
            continue

        if kind == "llm_result":
            current.result = record
            continue

        if kind != "message":
            continue

        message = record.get("message") or {}
        role = message.get("role")
        if role == "assistant":
            current.assistant = message
            current.thinking = _thinking_of(message)
            current.tool_calls = _tool_calls_of(message)
        elif role == "tool":
            current.tool_results.append(message)
        elif role in {"user", "system"}:
            text = _text_of(message)
            if text.lstrip().startswith(INJECTED_MARKERS):
                current.injected.append(text)
            else:
                current.user_messages.append(text)

    if current.request is not None or current.tool_calls or current.user_messages:
        turns.append(current)
    return Trace(turns=turns, events=events, model_id=model_id, provider=provider)


def export_trace(trace_dir: Path, output_dir: Path, *, model_source: str | None = None) -> Path:
    """Write the per-turn directory tree. Returns the `turns/` directory."""
    trace = read_trace(trace_dir)
    turns_dir = Path(output_dir) / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)

    for turn in trace.turns:
        directory = turns_dir / f"turn-{turn.number:03d}"
        directory.mkdir(parents=True, exist_ok=True)

        if turn.request:
            _write_json(directory / "request.json", turn.request)
        if turn.result:
            _write_json(directory / "response.json", turn.result)
        if turn.thinking:
            (directory / "thinking.txt").write_text(turn.thinking + "\n")
        if turn.assistant and (text := _text_of(turn.assistant)):
            (directory / "assistant.txt").write_text(text + "\n")
        if turn.tool_calls:
            _write_json(directory / "tool-calls.json", turn.tool_calls)
        if turn.injected:
            (directory / "injected.txt").write_text("\n\n".join(turn.injected) + "\n")
        if turn.user_messages:
            (directory / "user.txt").write_text("\n\n".join(turn.user_messages) + "\n")

        if turn.tool_results:
            results_dir = directory / "tool-results"
            results_dir.mkdir(exist_ok=True)
            for index, result in enumerate(turn.tool_results):
                name = result.get("name") or f"tool-{index}"
                (results_dir / f"{index:02d}-{name}.txt").write_text(_result_text(result))

        source = _source_after(turn)
        if source is not None:
            (directory / "model.py").write_text(source)

    if model_source is not None:
        (turns_dir.parent / "model-final.py").write_text(model_source)
    (turns_dir.parent / "turns.txt").write_text(trace.to_text() + "\n")
    return turns_dir


def _source_after(turn: Turn) -> str | None:
    """The whole file, when this turn's edit happened to contain it.

    A `write_file` call carries the finished text, so the file as it stood after that turn
    can be recovered exactly. A `replace` call carries only a fragment, and reconstructing
    the file from a chain of fragments would mean re-implementing the editor and getting a
    plausible answer wrong. Better to write nothing than to write a guess.
    """
    for call in turn.tool_calls:
        if call.get("name") in {"write_file", "create_file"}:
            arguments = call.get("arguments") or {}
            for key in ("content", "text", "file_text", "code"):
                if isinstance(arguments.get(key), str):
                    return arguments[key]
    return None


def _tool_calls_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        calls.append(
            {
                "id": raw.get("id"),
                "name": function.get("name") or raw.get("name"),
                "arguments": arguments,
            }
        )
    return calls


def _result_text(message: dict[str, Any]) -> str:
    """A tool result as text, with its payload unwrapped.

    Results arrive as a JSON string wrapped around another string that is usually a
    `<compile_signals>` or `<grounding_signals>` block. Unwrapping it is the difference
    between a readable file and a wall of escaped newlines.
    """
    content = message.get("content")
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return content
    else:
        payload = content

    if isinstance(payload, dict):
        for key in ("result", "error", "output"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if value is not None:
                return json.dumps(value, indent=2, ensure_ascii=False)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, dict) and item.get("type", "").endswith("image"):
                parts.append(f"[image: {item.get('name', 'attached')}]")
        return "\n".join(parts)
    return ""


def _thinking_of(message: dict[str, Any]) -> str:
    for key in ("thought_summary", "thinking", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    extra = message.get("extra_content")
    if isinstance(extra, dict):
        for payload in extra.values():
            if isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, str) and value.strip():
                        return value
    return ""


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
