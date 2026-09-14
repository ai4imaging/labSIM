"""
EditCode tool - Make precise edits to code files using old_string/new_string
"""

import aiofiles

from agent.batch_edit import (
    BATCHING_GUIDANCE,
    COMPILE_SEPARATELY_NOTE,
    EDITS_PARAMETER,
    SYNTAX_ONLY_NOTE,
    ReplaceEdit,
    apply_edits,
    resolve_edits,
)
from agent.tools.base import (
    BaseDeclarativeTool,
    BoundFileToolInvocation,
    ToolParamsModel,
    ToolResult,
    make_tool_schema,
    validate_tool_params,
)
from agent.tools.model_contract import missing_required_model_contract


def validate_python_syntax(full_code: str, filename: str) -> dict:
    """
    Validate Python syntax by attempting to parse the code.
    """
    try:
        compile(full_code, filename, "exec")
        return {
            "status": "success",
            "error": None,
        }
    except SyntaxError as e:
        error_msg = f"Syntax error: {e.msg} (line {e.lineno})"
        return {
            "status": "error",
            "error": error_msg,
            "error_line": e.lineno,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Validation error: {str(e)}",
        }


class EditCodeParams(ToolParamsModel):
    """Parameters for edit_code tool"""

    old_string: str
    new_string: str
    replace_all: bool = False


class EditCodeInvocation(BoundFileToolInvocation[EditCodeParams, str]):
    """Invocation for editing code file"""

    def get_description(self) -> str:
        preview = self.params.old_string[:50]
        if len(self.params.old_string) > 50:
            preview += "..."
        return f"Edit current target file: replacing '{preview}'"

    async def execute(self) -> ToolResult:
        try:
            if not self.file_path:
                return ToolResult(error="file_path is required")

            # Load current code
            async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as f:
                full_code = await f.read()

            if not self.params.old_string:
                if full_code.strip():
                    return ToolResult(
                        error=(
                            "old_string cannot be empty unless model.py is empty. "
                            "Please provide the exact string to replace."
                        )
                    )
                new_code = self.params.new_string
                missing = missing_required_model_contract(new_code)
                if missing:
                    return ToolResult(
                        error=(
                            "replace must initialize model.py with the complete top-level contract: "
                            + ", ".join(missing)
                        )
                    )
                validation = self._validate_python_syntax(new_code, self.file_path or "<string>")
                async with aiofiles.open(self.file_path, mode="w", encoding="utf-8") as f:
                    await f.write(new_code)
                return ToolResult(output="Code edited successfully", compilation=validation)

            # Check if old_string exists in the full model file.
            if self.params.old_string not in full_code:
                return ToolResult(
                    error="Could not find the old_string in the code. "
                    "Make sure the string matches exactly, including whitespace and indentation. "
                )

            # Count occurrences
            occurrences = full_code.count(self.params.old_string)
            if occurrences > 1 and not self.params.replace_all:
                return ToolResult(
                    error=f"The old_string appears {occurrences} times in the code. "
                    f"Please provide a longer, unique string that appears only once, "
                    f"or use replace_all=true to replace all occurrences."
                )

            # Perform replacement
            if self.params.replace_all:
                new_code = full_code.replace(self.params.old_string, self.params.new_string)
            else:
                new_code = full_code.replace(self.params.old_string, self.params.new_string, 1)

            missing = missing_required_model_contract(new_code)
            if missing:
                return ToolResult(
                    error=(
                        "replace must preserve the complete top-level model.py contract: "
                        + ", ".join(missing)
                    )
                )

            # Validate Python syntax
            validation = self._validate_python_syntax(new_code, self.file_path or "<string>")

            # Save updated code
            async with aiofiles.open(self.file_path, mode="w", encoding="utf-8") as f:
                await f.write(new_code)

            # Build success message
            if self.params.replace_all and occurrences > 1:
                success_msg = f"Code edited successfully (replaced {occurrences} occurrences)"
            else:
                success_msg = "Code edited successfully"

            return ToolResult(output=success_msg, compilation=validation)

        except FileNotFoundError:
            return ToolResult(error=f"File {self.file_path} not found")
        except Exception as e:
            return ToolResult(error=f"Error editing code: {str(e)}")

    def _validate_python_syntax(self, full_code: str, filename: str) -> dict:
        """
        Validate Python syntax by attempting to parse the code.
        """
        return validate_python_syntax(full_code, filename)


class ReplaceParams(ToolParamsModel):
    """Parameters for Gemini-style replace tool.

    `old_string`/`new_string` is the single-edit form and `edits` is the batch form. Both
    are optional at the schema level because exactly one is expected; `agent.batch_edit`
    decides which was meant.
    """

    old_string: str | None = None
    new_string: str | None = None
    edits: list[ReplaceEdit] | None = None
    instruction: str | None = None
    allow_multiple: bool = False

    def resolve_edits(self) -> tuple[list[ReplaceEdit], str | None]:
        return resolve_edits(
            old_string=self.old_string,
            new_string=self.new_string,
            edits=self.edits,
            allow_multiple=self.allow_multiple,
        )


class ReplaceInvocation(BoundFileToolInvocation[ReplaceParams, str]):
    """Invocation for full-file replacements under the official-style name."""

    def get_description(self) -> str:
        edits, _ = self.params.resolve_edits()
        if len(edits) > 1:
            return f"Apply {len(edits)} edits to current target file"
        preview = edits[0].old_string[:50] if edits else ""
        if len(preview) == 50:
            preview += "..."
        return f"Replace text in current target file: '{preview}'"

    async def execute(self) -> ToolResult:
        try:
            if not self.file_path:
                return ToolResult(error="file_path is required")

            edits, form_error = self.params.resolve_edits()
            if form_error:
                return ToolResult(error=form_error)

            async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as f:
                full_code = await f.read()

            new_code, failure = apply_edits(full_code, edits)
            if failure:
                return ToolResult(error=f"{failure} Nothing was written.")

            missing = missing_required_model_contract(new_code)
            if missing:
                return ToolResult(
                    error=(
                        "replace must preserve the complete top-level model.py contract: "
                        + ", ".join(missing)
                    )
                )

            validation = validate_python_syntax(new_code, self.file_path)
            async with aiofiles.open(self.file_path, mode="w", encoding="utf-8") as f:
                await f.write(new_code)

            if len(edits) > 1:
                applied = f"Applied {len(edits)} edits successfully"
            else:
                applied = "Code edited successfully"
            return ToolResult(output=f"{applied}. {SYNTAX_ONLY_NOTE}", compilation=validation)

        except FileNotFoundError:
            return ToolResult(error=f"File {self.file_path} not found")
        except Exception as e:
            return ToolResult(error=f"Error editing code: {str(e)}")


class ReplaceTool(BaseDeclarativeTool):
    """Gemini-style replace tool scoped to the full model file."""

    def __init__(self) -> None:
        schema = make_tool_schema(
            name="replace",
            description=(
                "Replace text within the current bound `model.py` file.\n\n"
                "This is the Gemini-style edit tool for the bound `model.py` artifact.\n"
                "It operates on the full visible model script, including imports and the object_model footer.\n\n"
                f"{BATCHING_GUIDANCE}\n"
                "Matching behavior:\n"
                "- `old_string` must match EXACTLY, including whitespace, indentation, and newlines\n"
                "- If `model.py` is empty, `old_string` may be empty to insert the initial code\n"
                "- By default, `old_string` must appear exactly once\n"
                "- Set `allow_multiple=true` to replace all exact matches\n"
                '- If exact matching fails, reread `read_file(path="model.py")` and retry with a smaller exact snippet\n\n'
                f"{COMPILE_SEPARATELY_NOTE}"
            ),
            parameters={
                "old_string": {
                    "type": "string",
                    "description": (
                        "Exact literal text to find inside the full model.py file. "
                        "Must match including whitespace and indentation. "
                        "Omit when using `edits`."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": (
                        "Replacement text. May be empty to delete the matched text. "
                        "Omit when using `edits`."
                    ),
                },
                "edits": EDITS_PARAMETER,
                "instruction": {
                    "type": "string",
                    "description": (
                        "Optional short description of the intended change. "
                        "Accepted for official Gemini CLI parity."
                    ),
                },
                "allow_multiple": {
                    "type": "boolean",
                    "description": (
                        "If false (default), `old_string` must be unique. "
                        "If true, replace all exact matches. Applies to the single-edit form."
                    ),
                },
            },
            required=[],
        )
        super().__init__("replace", schema)

    async def build(self, params: dict) -> ReplaceInvocation:
        validated = validate_tool_params(ReplaceParams, params)
        return ReplaceInvocation(validated)
