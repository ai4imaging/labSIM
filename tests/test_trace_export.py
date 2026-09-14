"""Trace expansion, checked against a hand-written trajectory.

The interesting cases are the ones where the flat log is ambiguous: telling a harness
reminder apart from the task prompt, unwrapping a tool result that is JSON inside JSON,
and declining to reconstruct a file from a fragment. A synthetic trace pins all three
down without needing an LLM run to produce one.
"""

from __future__ import annotations

import json

from amx.trace_export import export_trace, read_trace


def _write(tmp_path, records):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "trajectory.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    return trace_dir


def _request(turn, tools=("compile_model", "check_physical_grounding")):
    return {
        "ts": 1.0,
        "type": "llm_request",
        "turn": turn,
        "provider": "gpugeek",
        "model_id": "Vendor2/Claude-4.5-Sonnet",
        "message_count": 4,
        "tool_names": list(tools),
        "system_prompt_chars": 12000,
    }


def _result(turn, calls=("compile_model",)):
    return {
        "ts": 2.0,
        "type": "llm_result",
        "turn": turn,
        "duration_s": 3.5,
        "text_chars": 120,
        "tool_calls": list(calls),
        "usage": {"total_tokens": 4200},
    }


def _assistant(name="compile_model", arguments="{}"):
    return {
        "ts": 2.1,
        "type": "message",
        "message": {
            "role": "assistant",
            "content": "Compiling now.",
            "tool_calls": [
                {"id": "c1", "function": {"name": name, "arguments": arguments}}
            ],
        },
    }


def _tool_result(name="compile_model", payload="<compile_signals>\nall good\n</compile_signals>"):
    return {
        "ts": 2.2,
        "type": "message",
        "message": {
            "role": "tool",
            "tool_call_id": "c1",
            "name": name,
            "content": json.dumps({"result": payload}),
        },
    }


def test_turns_are_split_on_the_request_event(tmp_path):
    trace_dir = _write(
        tmp_path,
        [
            _request(1), _assistant(), _tool_result(), _result(1),
            _request(2), _assistant(), _tool_result(), _result(2),
        ],
    )
    trace = read_trace(trace_dir)
    assert [turn.number for turn in trace.turns] == [1, 2]
    assert trace.model_id == "Vendor2/Claude-4.5-Sonnet"
    assert trace.provider == "gpugeek"


def test_harness_reminders_are_kept_apart_from_the_task(tmp_path):
    trace_dir = _write(
        tmp_path,
        [
            _request(1),
            {
                "ts": 1.1, "type": "message",
                "message": {"role": "user", "content": "Build a 250 mL beaker."},
            },
            {
                "ts": 1.2, "type": "message",
                "message": {
                    "role": "user",
                    "content": "<grounding_required>\nThe code compiles, but...\n</grounding_required>",
                },
            },
            _result(1),
        ],
    )
    turn = read_trace(trace_dir).turns[0]
    assert turn.user_messages == ["Build a 250 mL beaker."]
    assert len(turn.injected) == 1 and "grounding_required" in turn.injected[0]


def test_a_tool_result_is_unwrapped_rather_than_left_as_escaped_json(tmp_path):
    trace_dir = _write(
        tmp_path,
        [_request(1), _assistant(), _tool_result(), _result(1)],
    )
    export_trace(trace_dir, tmp_path / "out")
    written = (
        tmp_path / "out" / "turns" / "turn-001" / "tool-results" / "00-compile_model.txt"
    ).read_text()
    assert written.startswith("<compile_signals>")
    assert "\\n" not in written


def test_tool_call_arguments_are_parsed(tmp_path):
    trace_dir = _write(
        tmp_path,
        [
            _request(1),
            _assistant(name="replace", arguments='{"old_string": "a", "new_string": "b"}'),
            _result(1, calls=("replace",)),
        ],
    )
    export_trace(trace_dir, tmp_path / "out")
    calls = json.loads((tmp_path / "out" / "turns" / "turn-001" / "tool-calls.json").read_text())
    assert calls[0]["arguments"] == {"old_string": "a", "new_string": "b"}


def test_a_whole_file_write_is_recovered_and_a_fragment_is_not(tmp_path):
    """A `replace` carries a fragment; guessing the file around it would be fiction."""
    trace_dir = _write(
        tmp_path,
        [
            _request(1),
            _assistant(name="write_file", arguments=json.dumps({"content": "radius = 0.035\n"})),
            _result(1, calls=("write_file",)),
            _request(2),
            _assistant(name="replace", arguments='{"old_string": "0.035", "new_string": "0.036"}'),
            _result(2, calls=("replace",)),
        ],
    )
    export_trace(trace_dir, tmp_path / "out")
    turns = tmp_path / "out" / "turns"
    assert (turns / "turn-001" / "model.py").read_text() == "radius = 0.035\n"
    assert not (turns / "turn-002" / "model.py").exists()


def test_the_summary_lists_every_turn_with_its_tools(tmp_path):
    trace_dir = _write(
        tmp_path,
        [_request(1), _assistant(), _tool_result(), _result(1)],
    )
    export_trace(trace_dir, tmp_path / "out")
    summary = (tmp_path / "out" / "turns.txt").read_text()
    assert "turn   1" in summary and "compile_model" in summary and "4200 tok" in summary


def test_an_image_attachment_is_noted_rather_than_dumped(tmp_path):
    trace_dir = _write(
        tmp_path,
        [
            _request(1),
            {
                "ts": 1.1, "type": "message",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "<grounding_renders>views</grounding_renders>"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "name": "front"},
                    ],
                },
            },
            _result(1),
        ],
    )
    turn = read_trace(trace_dir).turns[0]
    assert turn.injected and "[image: front]" in turn.injected[0]
    assert "AAAA" not in turn.injected[0]
