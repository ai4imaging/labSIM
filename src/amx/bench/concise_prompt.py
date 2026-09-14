"""Reduce a benchmark specification to the one paragraph a plain baseline sees."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from amx.bench.case import BenchCase
from amx.paths import activate_articraft


def concise_prompt(
    case: BenchCase,
    case_dir: Path,
    *,
    provider: str,
    model_id: str | None,
    thinking_level: str,
) -> str:
    """Generate and cache one prose paragraph from input.md, never from the rubric."""
    case_dir = Path(case_dir)
    path = case_dir / "concise-prompt.md"
    if path.is_file() and (text := path.read_text().strip()):
        return text

    method = "llm"
    try:
        text = asyncio.run(
            _summarize(
                case.input_path.read_text(),
                provider=provider,
                model_id=model_id,
                thinking_level=thinking_level,
            )
        )
    except Exception as error:  # noqa: BLE001 - a baseline should survive summarizer downtime
        text = _deterministic_fallback(case)
        method = f"deterministic-fallback: {type(error).__name__}: {error}"

    text = _one_paragraph(text)
    case_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    (case_dir / "concise-prompt-meta.json").write_text(
        json.dumps(
            {
                "source": "input.md only",
                "method": method,
                "provider": provider,
                "model_id": model_id,
                "characters": len(text),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return text


async def _summarize(
    input_text: str,
    *,
    provider: str,
    model_id: str | None,
    thinking_level: str,
) -> str:
    if provider.lower() != "gpugeek":
        raise ValueError("AI concise-prompt generation currently requires gpugeek")
    activate_articraft()
    from agent.providers.gpugeek import GpuGeekLLM  # noqa: PLC0415

    client = GpuGeekLLM(model_id=model_id, thinking_level=thinking_level)
    try:
        response = await client.generate_with_tools(
            (
                "Convert the supplied asset specification into one concise English paragraph "
                "for a generic 3D articulated-asset authoring agent. Preserve object identity, "
                "known overall dimensions, major visible components, and real movable mechanisms. "
                "Omit requirement IDs, provenance, confidence classes, rubrics, tests, scores, "
                "implementation instructions, headings, and bullet lists. Use 80–140 words."
            ),
            [{"role": "user", "content": input_text}],
            [],
        )
    finally:
        await client.close()
    content = response.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("summarizer returned no text")
    return content


def _deterministic_fallback(case: BenchCase) -> str:
    spec = case.to_grounding_spec()
    dimensions = ", ".join(
        f"{target.name.replace('_', ' ')} {target.value_m * 1000:g} mm"
        for target in spec.dimensions[:6]
    )
    components = ", ".join(component.name for component in spec.components[:10])
    operations = ", ".join(operation.name for operation in spec.operations[:8])
    sentences = [f"Create a simulation-ready 3D asset of {spec.summary or case.asset_class}."]
    if dimensions:
        sentences.append(f"Use these known dimensions: {dimensions}.")
    if components:
        sentences.append(f"Include the main components: {components}.")
    if operations:
        sentences.append(f"Make these real movable mechanisms rather than decoration: {operations}.")
    sentences.append("Use realistic proportions, materials, collision geometry, mass and inertia.")
    return " ".join(sentences)


def _one_paragraph(text: str) -> str:
    text = re.sub(r"^```(?:text)?|```$", "", text.strip(), flags=re.I)
    text = re.sub(r"^\s*(?:[-*#]+|\d+[.)])\s*", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()
