from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")

BENCHMARK_EXCLUSION_FILE = Path(
    "benchmarks/imo64/excluded_source_ids.json"
)

BENCHMARK_QUESTIONS_FILE = Path(
    "benchmarks/imo64/questions.jsonl"
)

OUTPUT_DIR = Path("data/training")

POOL_OUTPUT = OUTPUT_DIR / "imo_pool.jsonl"
SFT_OUTPUT = OUTPUT_DIR / "imo_sft.jsonl"
MANIFEST_OUTPUT = OUTPUT_DIR / "imo_manifest.json"


SYSTEM_PROMPT = (
    "You are an expert mathematical reasoner and olympiad solver. "
    "Solve the problem rigorously. Explain the important ideas clearly, "
    "justify every nontrivial step, and use LaTeX for mathematical notation."
)


# ============================================================
# FILE HELPERS
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            f.write("\n")


# ============================================================
# NORMALIZATION / HASHING
# ============================================================

def normalize_text(text: str) -> str:
    """
    Simple normalization used for leakage checking.

    This is not yet our full fuzzy deduplication system.
    We will make a stronger one before importing large
    Hugging Face datasets.
    """

    text = text.lower()

    text = "".join(
        text.split()
    )

    return text


def statement_hash(
    statement: str,
) -> str:

    normalized = normalize_text(
        statement
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# LOAD BENCHMARK PROTECTION
# ============================================================

def load_benchmark_exclusions():

    if not BENCHMARK_EXCLUSION_FILE.exists():

        raise FileNotFoundError(
            f"Missing benchmark exclusion file:\n"
            f"{BENCHMARK_EXCLUSION_FILE}\n\n"
            "Run build_imo64.py first."
        )

    exclusion_data = load_json(
        BENCHMARK_EXCLUSION_FILE
    )

    excluded_ids = {
        str(problem_id)
        for problem_id
        in exclusion_data["source_ids"]
    }

    if len(excluded_ids) != 64:

        raise ValueError(
            f"Expected exactly 64 benchmark IDs, "
            f"found {len(excluded_ids)}."
        )

    return excluded_ids


def load_benchmark_statement_hashes():

    if not BENCHMARK_QUESTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Missing benchmark questions:\n"
            f"{BENCHMARK_QUESTIONS_FILE}"
        )

    hashes = set()

    with BENCHMARK_QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            hashes.add(
                statement_hash(
                    record["statement"]
                )
            )

    if len(hashes) != 64:

        raise ValueError(
            "Expected 64 unique benchmark statements, "
            f"found {len(hashes)}."
        )

    return hashes


# ============================================================
# LOAD THE 425 PROCESSED IMO PROBLEMS
# ============================================================

def load_processed_problems():

    files = sorted(
        PROCESSED_DIR.glob(
            "problem_*.json"
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No problems found in {PROCESSED_DIR}"
        )

    problems = []

    required_fields = {
        "id",
        "source",
        "year",
        "topic",
        "subtopics",
        "statement",
        "solutions",
    }

    for path in files:

        problem = load_json(path)

        missing = (
            required_fields
            - problem.keys()
        )

        if missing:

            raise ValueError(
                f"{path.name} is missing fields: "
                f"{sorted(missing)}"
            )

        if not str(
            problem["statement"]
        ).strip():

            raise ValueError(
                f"{path.name} has no statement."
            )

        if not problem["solutions"]:

            raise ValueError(
                f"{path.name} has no solutions."
            )

        problem["_source_file"] = (
            path.name
        )

        problems.append(problem)

    return problems


# ============================================================
# REMOVE THE 64 BENCHMARK PROBLEMS
# ============================================================

def create_imo_training_pool(
    problems,
    excluded_ids,
    benchmark_hashes,
):

    training = []

    skipped_by_id = []
    skipped_by_hash = []

    for problem in problems:

        source_id = str(
            problem["id"]
        )

        current_hash = statement_hash(
            problem["statement"]
        )

        # First protection:
        # benchmark source ID.
        if source_id in excluded_ids:

            skipped_by_id.append(
                source_id
            )

            continue

        # Second protection:
        # exact normalized statement duplicate.
        if current_hash in benchmark_hashes:

            skipped_by_hash.append(
                source_id
            )

            continue

        training.append(problem)

    return (
        training,
        skipped_by_id,
        skipped_by_hash,
    )


# ============================================================
# BUILD ONE-PROBLEM-PER-ROW MASTER POOL
# ============================================================

def build_pool_rows(
    problems,
):

    rows = []

    for problem in problems:

        row = {
            "id": str(
                problem["id"]
            ),

            "source": problem[
                "source"
            ],

            "year": str(
                problem["year"]
            ),

            "topic": problem[
                "topic"
            ],

            "subtopics": problem[
                "subtopics"
            ],

            "statement": problem[
                "statement"
            ],

            "solutions": problem[
                "solutions"
            ],

            "statement_sha256":
                statement_hash(
                    problem["statement"]
                ),
        }

        if "problem_code" in problem:

            row["problem_code"] = (
                problem["problem_code"]
            )

        rows.append(row)

    return rows


# ============================================================
# BUILD SUPERVISED FINE-TUNING FORMAT
#
# One problem with two official solutions becomes
# TWO SFT examples.
#
# That teaches the model multiple legitimate approaches.
# ============================================================

def build_sft_rows(
    problems,
):

    rows = []

    for problem in problems:

        source_id = str(
            problem["id"]
        )

        solutions = problem[
            "solutions"
        ]

        for solution_index, solution in enumerate(
            solutions,
            start=1,
        ):

            if not str(
                solution
            ).strip():

                continue

            row = {

                "messages": [
                    {
                        "role": "system",
                        "content":
                            SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content":
                            problem[
                                "statement"
                            ],
                    },
                    {
                        "role": "assistant",
                        "content":
                            solution,
                    },
                ],

                "metadata": {
                    "source":
                        problem[
                            "source"
                        ],

                    "source_id":
                        source_id,

                    "year":
                        str(
                            problem[
                                "year"
                            ]
                        ),

                    "topic":
                        problem[
                            "topic"
                        ],

                    "subtopics":
                        problem[
                            "subtopics"
                        ],

                    "solution_index":
                        solution_index,

                    "statement_sha256":
                        statement_hash(
                            problem[
                                "statement"
                            ]
                        ),
                },
            }

            if "problem_code" in problem:

                row["metadata"][
                    "problem_code"
                ] = problem[
                    "problem_code"
                ]

            rows.append(row)

    return rows


# ============================================================
# MANIFEST
# ============================================================

def build_manifest(
    all_problems,
    training_problems,
    sft_rows,
    skipped_by_id,
    skipped_by_hash,
):

    topic_counts = Counter(
        problem["topic"]
        for problem
        in training_problems
    )

    year_counts = Counter(
        str(problem["year"])
        for problem
        in training_problems
    )

    return {

        "dataset_name":
            "OlympiadMathAI IMO Training Pool",

        "source":
            "IMO Shortlist",

        "input_processed_problems":
            len(all_problems),

        "protected_benchmark_problems":
            len(
                set(skipped_by_id)
            ),

        "additional_hash_exclusions":
            len(
                set(skipped_by_hash)
            ),

        "training_problems":
            len(training_problems),

        "sft_examples":
            len(sft_rows),

        "topics":
            dict(
                sorted(
                    topic_counts.items()
                )
            ),

        "years":
            dict(
                sorted(
                    year_counts.items()
                )
            ),

        "benchmark_protection": {
            "id_exclusion":
                True,

            "normalized_statement_hash":
                True,

            "benchmark":
                "IMO64",
        },

        "files": {
            "master_pool":
                str(POOL_OUTPUT),

            "sft":
                str(SFT_OUTPUT),
        },
    }


# ============================================================
# FINAL SAFETY CHECK
# ============================================================

def verify_no_benchmark_leakage(
    training_problems,
    excluded_ids,
    benchmark_hashes,
):

    for problem in training_problems:

        source_id = str(
            problem["id"]
        )

        if source_id in excluded_ids:

            raise RuntimeError(
                "BENCHMARK LEAK DETECTED: "
                f"source ID {source_id}"
            )

        current_hash = statement_hash(
            problem["statement"]
        )

        if current_hash in benchmark_hashes:

            raise RuntimeError(
                "BENCHMARK LEAK DETECTED "
                "BY STATEMENT HASH: "
                f"{source_id}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("BUILDING IMO TRAINING DATA")
    print("=" * 65)
    print()

    excluded_ids = (
        load_benchmark_exclusions()
    )

    benchmark_hashes = (
        load_benchmark_statement_hashes()
    )

    problems = (
        load_processed_problems()
    )

    print(
        f"Loaded processed problems: "
        f"{len(problems)}"
    )

    (
        training_problems,
        skipped_by_id,
        skipped_by_hash,
    ) = create_imo_training_pool(
        problems,
        excluded_ids,
        benchmark_hashes,
    )

    verify_no_benchmark_leakage(
        training_problems,
        excluded_ids,
        benchmark_hashes,
    )

    pool_rows = build_pool_rows(
        training_problems
    )

    sft_rows = build_sft_rows(
        training_problems
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        POOL_OUTPUT,
        pool_rows,
    )

    write_jsonl(
        SFT_OUTPUT,
        sft_rows,
    )

    manifest = build_manifest(
        problems,
        training_problems,
        sft_rows,
        skipped_by_id,
        skipped_by_hash,
    )

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    print()
    print("Benchmark protection:")
    print(
        f"  Excluded by ID: "
        f"{len(skipped_by_id)}"
    )
    print(
        f"  Extra excluded by hash: "
        f"{len(skipped_by_hash)}"
    )

    print()
    print(
        f"IMO training problems: "
        f"{len(training_problems)}"
    )

    print(
        f"SFT examples: "
        f"{len(sft_rows)}"
    )

    print()
    print("Created:")

    print(
        f"  {POOL_OUTPUT}"
    )

    print(
        f"  {SFT_OUTPUT}"
    )

    print(
        f"  {MANIFEST_OUTPUT}"
    )

    print()

    if (
        len(problems) == 425
        and len(training_problems)
        != 361
    ):

        print(
            "WARNING: You started with 425 "
            "problems, so normally 361 should "
            "remain after excluding IMO64."
        )

    else:

        print(
            "Benchmark leakage check: PASSED"
        )

    print()
    print("=" * 65)


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("=" * 65)
        print("ERROR")
        print("=" * 65)
        print()
        print(error)
        print()

        sys.exit(1)