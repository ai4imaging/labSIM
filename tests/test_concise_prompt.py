from pathlib import Path

from amx.bench.case import load_case
from amx.bench.concise_prompt import concise_prompt

CASES = Path(__file__).resolve().parents[1] / "3D_asset_cases"


def test_concise_prompt_is_one_cached_paragraph_from_input_only(tmp_path, monkeypatch):
    case = load_case(CASES, "CEN-001")
    calls = 0

    async def fake_summarize(input_text: str, **_kwargs) -> str:
        nonlocal calls
        calls += 1
        assert case.input_path.read_text() == input_text
        return "# Asset\n- Build a centrifuge.\n- Include a lid and spinning rotor."

    monkeypatch.setattr("amx.bench.concise_prompt._summarize", fake_summarize)
    first = concise_prompt(
        case,
        tmp_path,
        provider="gpugeek",
        model_id="Vendor2/Claude-4.8-opus",
        thinking_level="medium",
    )
    second = concise_prompt(
        case,
        tmp_path,
        provider="gpugeek",
        model_id="Vendor2/Claude-4.8-opus",
        thinking_level="medium",
    )

    assert first == second == "Asset Build a centrifuge. Include a lid and spinning rotor."
    assert "\n" not in first
    assert calls == 1
    assert "rubric" not in first.lower()
