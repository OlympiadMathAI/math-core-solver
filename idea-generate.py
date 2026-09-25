import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load GEMINI_API_KEY from .env
load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found.\n"
        "Create a .env file in the project root containing:\n"
        "GEMINI_API_KEY=your_actual_api_key"
    )


MODEL_CANDIDATES = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
)

BEGIN_MARKER = "<<<BEGIN_SOLUTION>>>"
END_MARKER = "<<<END_SOLUTION>>>"


SYSTEM_INSTRUCTION = f"""
You are an expert mathematical olympiad solver.

Return one complete, polished, self-contained proof.

Requirements:
- Start the solution from the beginning.
- State all definitions and constructions that are needed.
- Justify every important step rigorously.
- Use LaTeX for mathematical expressions.
- Do not output scratch work, planning, internal analysis, self-evaluation,
  or statements such as "I will now write the proof."
- Do not begin in the middle of an argument.
- Do not refer to an earlier response.
- Do not place the answer inside a Markdown code block.
- End only after the proof is fully complete.

Begin your response with exactly:
{BEGIN_MARKER}

End your response with exactly:
{END_MARKER}
""".strip()


class IncompleteResponseError(RuntimeError):
    """Raised when Gemini returns only part of a solution."""

def get_exception_chain_text(error: BaseException) -> str:
    """
    Collect text from an exception and its underlying causes.

    The Interactions API may wrap the original server error inside
    another SDK exception, so we inspect the complete exception chain.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(
            f"{type(current).__module__}."
            f"{type(current).__name__}: {current}"
        )

        current = current.__cause__ or current.__context__

    return " | ".join(parts).lower()


def classify_gemini_error(error: BaseException) -> str:
    """
    Classify Gemini errors without depending on private SDK classes.

    Returns:
        "temporary"         - retry or try another model
        "model_unavailable" - try another model
        "quota"             - quota must reset or be increased
        "fatal"             - invalid key, permission problem, bad request, etc.
    """
    text = get_exception_chain_text(error)

    if "quota_exceeded" in text or "daily quota" in text:
        return "quota"

    temporary_markers = (
        "service_unavailable",
        "high demand",
        "503",
        "rate_limit_exceeded",
        "too_many_requests",
        "429",
        "deadline_exceeded",
        "504",
        "bad gateway",
        "502",
        "internal server error",
        "internalservererror",
        "api_error",
    )

    if any(marker in text for marker in temporary_markers):
        return "temporary"

    model_markers = (
        "model_not_found",
        "model not found",
        "404",
    )

    if any(marker in text for marker in model_markers):
        return "model_unavailable"

    return "fatal"

def extract_complete_solution(raw_text: str) -> str:
    """
    Extract the final proof between the required markers.

    This prevents incomplete responses or model planning text
    from being shown as the final solution.
    """
    start = raw_text.find(BEGIN_MARKER)
    end = raw_text.find(END_MARKER)

    if start == -1 or end == -1 or end <= start:
        raise IncompleteResponseError(
            "The response did not contain both completion markers."
        )

    solution = raw_text[
        start + len(BEGIN_MARKER):end
    ].strip()

    if len(solution) < 100:
        raise IncompleteResponseError(
            "The returned solution was suspiciously short."
        )

    return solution


def request_solution(
    client: genai.Client,
    model_name: str,
    problem_statement: str,
    retry: bool = False,
) -> str:
    """Request one complete solution from one Gemini model."""

    retry_instruction = ""

    if retry:
        retry_instruction = """
A previous attempt returned an incomplete or malformed response.
Restart completely from the beginning. Do not continue the previous
attempt and do not output any planning text.
"""

    prompt = f"""
Solve the following olympiad problem.

{problem_statement}

{retry_instruction}
""".strip()

    interaction = client.interactions.create(
        model=model_name,
        input=prompt,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={
            # High reasoning quality for olympiad mathematics.
            # We intentionally do not set max_output_tokens.
            "thinking_level": "high",
        },
    )

    raw_text = interaction.output_text or ""

    # Save the exact raw response for debugging.
    Path("debug_last_response.txt").write_text(
        raw_text,
        encoding="utf-8",
    )

    return extract_complete_solution(raw_text)


def generate_math_solution(problem_statement: str) -> str:
    """Try several Gemini models until one returns a complete proof."""

    problem_statement = problem_statement.strip()

    if not problem_statement:
        raise ValueError("The problem statement cannot be empty.")

    last_error: Exception | None = None
    max_attempts_per_model = 2

    with genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(timeout=180_000),
    ) as client:

        for model_name in MODEL_CANDIDATES:
            print(f"Trying {model_name}...")

            retry_incomplete_response = False

            for attempt in range(max_attempts_per_model):
                try:
                    solution = request_solution(
                        client=client,
                        model_name=model_name,
                        problem_statement=problem_statement,
                        retry=retry_incomplete_response,
                    )

                    print(
                        f"Complete solution generated with "
                        f"{model_name}."
                    )

                    return solution

                except IncompleteResponseError as error:
                    last_error = error
                    retry_incomplete_response = True

                    if attempt < max_attempts_per_model - 1:
                        print(
                            f"{model_name} returned an incomplete "
                            "response. Retrying from the beginning..."
                        )
                        continue

                    print(
                        f"{model_name} returned another incomplete "
                        "response. Trying the next model..."
                    )
                    break

                except Exception as error:
                    # Interactions API errors may not inherit from
                    # google.genai.errors.APIError.
                    last_error = error
                    error_kind = classify_gemini_error(error)

                    if error_kind == "temporary":
                        if attempt < max_attempts_per_model - 1:
                            delay = 5 * (2 ** attempt)

                            print(
                                f"{model_name} is temporarily busy. "
                                f"Retrying in {delay} seconds..."
                            )

                            time.sleep(delay)
                            continue

                        print(
                            f"{model_name} is still busy. "
                            "Trying the next model..."
                        )
                        break

                    if error_kind == "model_unavailable":
                        print(
                            f"{model_name} is unavailable for this "
                            "API key. Trying the next model..."
                        )
                        break

                    if error_kind == "quota":
                        raise RuntimeError(
                            "The Gemini API quota has been exhausted. "
                            "Wait for the quota to reset or check the "
                            "project's usage limits."
                        ) from error

                    # Do not hide permanent or unexpected errors.
                    error_details = get_exception_chain_text(error)

                    raise RuntimeError(
                        "Gemini returned a non-retryable error:\n"
                        f"{error_details}"
                    ) from error

    raise RuntimeError(
        "Every candidate model was busy, unavailable, or returned "
        "an incomplete solution. Wait briefly and run the program again."
    ) from last_error

def read_multiline_problem() -> str:
    """
    Read a problem that may occupy multiple lines.

    The user finishes input by entering END on a separate line.
    """
    print("Paste the full problem below.")
    print("When finished, type END on a separate line.\n")

    lines: list[str] = []

    while True:
        line = input()

        if line.strip() == "END":
            break

        lines.append(line)

    return "\n".join(lines).strip()


def main() -> None:
    problem = read_multiline_problem()

    print("\n--- Generating solution ---")

    try:
        solution = generate_math_solution(problem)
    except (RuntimeError, ValueError) as error:
        print(f"\nERROR: {error}")
        return

    print("\n--- Complete Solution ---\n")
    print(solution)


if __name__ == "__main__":
    main()