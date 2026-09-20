from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("benchmarks/imo64")


# ============================================================
# SUBJECT CODES
# ============================================================

TOPIC_TO_PREFIX = {
    "Algebra": "A",
    "Combinatorics": "C",
    "Geometry": "G",
    "Number Theory": "N",
}

PREFIX_TO_TOPIC = {
    "A": "Algebra",
    "C": "Combinatorics",
    "G": "Geometry",
    "N": "Number Theory",
}


# ============================================================
# BENCHMARK DEFINITION
#
# 64 problems total:
# 16 Algebra
# 16 Combinatorics
# 16 Geometry
# 16 Number Theory
#
# L1-L4 are OUR benchmark difficulty bands.
# They are NOT official IMO difficulty ratings.
# ============================================================

@dataclass(frozen=True)
class BenchmarkSpec:
    year: int
    code: str
    level: str


BENCHMARK_SPECS = [

    # ========================================================
    # ALGEBRA
    # ========================================================

    # L1
    BenchmarkSpec(2010, "A1", "L1"),
    BenchmarkSpec(2014, "A1", "L1"),
    BenchmarkSpec(2018, "A1", "L1"),
    BenchmarkSpec(2022, "A1", "L1"),

    # L2
    BenchmarkSpec(2011, "A2", "L2"),
    BenchmarkSpec(2015, "A2", "L2"),
    BenchmarkSpec(2019, "A2", "L2"),
    BenchmarkSpec(2023, "A2", "L2"),

    # L3
    BenchmarkSpec(2012, "A4", "L3"),
    BenchmarkSpec(2016, "A4", "L3"),
    BenchmarkSpec(2020, "A4", "L3"),
    BenchmarkSpec(2021, "A4", "L3"),

    # L4
    BenchmarkSpec(2013, "A6", "L4"),
    BenchmarkSpec(2017, "A7", "L4"),
    BenchmarkSpec(2020, "A8", "L4"),
    BenchmarkSpec(2023, "A6", "L4"),


    # ========================================================
    # COMBINATORICS
    # ========================================================

    # L1
    BenchmarkSpec(2010, "C1", "L1"),
    BenchmarkSpec(2014, "C1", "L1"),
    BenchmarkSpec(2018, "C1", "L1"),
    BenchmarkSpec(2022, "C1", "L1"),

    # L2
    BenchmarkSpec(2011, "C2", "L2"),
    BenchmarkSpec(2015, "C2", "L2"),
    BenchmarkSpec(2019, "C2", "L2"),
    BenchmarkSpec(2023, "C2", "L2"),

    # L3
    BenchmarkSpec(2012, "C4", "L3"),
    BenchmarkSpec(2016, "C4", "L3"),
    BenchmarkSpec(2020, "C4", "L3"),
    BenchmarkSpec(2021, "C4", "L3"),

    # L4
    BenchmarkSpec(2013, "C6", "L4"),
    BenchmarkSpec(2017, "C6", "L4"),
    BenchmarkSpec(2019, "C6", "L4"),
    BenchmarkSpec(2023, "C6", "L4"),


    # ========================================================
    # GEOMETRY
    # ========================================================

    # L1
    BenchmarkSpec(2010, "G1", "L1"),
    BenchmarkSpec(2014, "G1", "L1"),
    BenchmarkSpec(2018, "G1", "L1"),
    BenchmarkSpec(2022, "G1", "L1"),

    # L2
    BenchmarkSpec(2011, "G2", "L2"),
    BenchmarkSpec(2015, "G2", "L2"),
    BenchmarkSpec(2019, "G2", "L2"),
    BenchmarkSpec(2023, "G2", "L2"),

    # L3
    BenchmarkSpec(2012, "G4", "L3"),
    BenchmarkSpec(2016, "G4", "L3"),
    BenchmarkSpec(2020, "G4", "L3"),
    BenchmarkSpec(2021, "G4", "L3"),

    # L4
    BenchmarkSpec(2013, "G6", "L4"),
    BenchmarkSpec(2017, "G6", "L4"),
    BenchmarkSpec(2019, "G6", "L4"),
    BenchmarkSpec(2023, "G6", "L4"),


    # ========================================================
    # NUMBER THEORY
    # ========================================================

    # L1
    BenchmarkSpec(2010, "N1", "L1"),
    BenchmarkSpec(2014, "N1", "L1"),
    BenchmarkSpec(2018, "N1", "L1"),
    BenchmarkSpec(2022, "N1", "L1"),

    # L2
    BenchmarkSpec(2011, "N2", "L2"),
    BenchmarkSpec(2015, "N2", "L2"),
    BenchmarkSpec(2019, "N2", "L2"),
    BenchmarkSpec(2023, "N2", "L2"),

    # L3
    BenchmarkSpec(2012, "N4", "L3"),
    BenchmarkSpec(2016, "N4", "L3"),
    BenchmarkSpec(2020, "N4", "L3"),
    BenchmarkSpec(2021, "N4", "L3"),

    # L4
    BenchmarkSpec(2013, "N6", "L4"),
    BenchmarkSpec(2017, "N6", "L4"),
    BenchmarkSpec(2019, "N6", "L4"),
    BenchmarkSpec(2023, "N6", "L4"),
]


# ============================================================
# BASIC FILE FUNCTIONS
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


# ============================================================
# HELPERS
# ============================================================

def numeric_id(problem: dict[str, Any]) -> int:
    try:
        return int(str(problem["id"]))
    except Exception as exc:
        raise ValueError(
            f"Problem has invalid id: {problem.get('id')}"
        ) from exc


def parse_problem_code(code: str) -> tuple[str, int]:

    if len(code) < 2:
        raise ValueError(f"Invalid problem code: {code}")

    prefix = code[0].upper()

    if prefix not in PREFIX_TO_TOPIC:
        raise ValueError(
            f"Unknown subject prefix in {code}"
        )

    try:
        number = int(code[1:])
    except ValueError as exc:
        raise ValueError(
            f"Invalid problem code: {code}"
        ) from exc

    return prefix, number


# ============================================================
# VALIDATE BENCHMARK CONFIG
# ============================================================

def validate_benchmark_specs() -> None:

    if len(BENCHMARK_SPECS) != 64:
        raise ValueError(
            f"Benchmark should contain 64 problems, "
            f"but contains {len(BENCHMARK_SPECS)}."
        )

    keys = [
        (x.year, x.code)
        for x in BENCHMARK_SPECS
    ]

    if len(keys) != len(set(keys)):

        duplicates = [
            key
            for key, count in Counter(keys).items()
            if count > 1
        ]

        raise ValueError(
            f"Duplicate benchmark problems: {duplicates}"
        )

    subject_counts = Counter()
    subject_level_counts = Counter()

    for spec in BENCHMARK_SPECS:

        prefix, _ = parse_problem_code(spec.code)

        topic = PREFIX_TO_TOPIC[prefix]

        subject_counts[topic] += 1
        subject_level_counts[(topic, spec.level)] += 1

        if spec.level not in {
            "L1",
            "L2",
            "L3",
            "L4",
        }:
            raise ValueError(
                f"Invalid difficulty level: {spec.level}"
            )

    for topic in TOPIC_TO_PREFIX:

        if subject_counts[topic] != 16:
            raise ValueError(
                f"{topic} has {subject_counts[topic]} problems, "
                f"expected 16."
            )

        for level in [
            "L1",
            "L2",
            "L3",
            "L4",
        ]:

            count = subject_level_counts[
                (topic, level)
            ]

            if count != 4:
                raise ValueError(
                    f"{topic} {level} has {count} problems, "
                    f"expected 4."
                )


# ============================================================
# LOAD YOUR EXISTING DATASET
# ============================================================

def load_processed_dataset(
    processed_dir: Path,
) -> list[dict[str, Any]]:

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {processed_dir}"
        )

    files = sorted(
        processed_dir.glob("problem_*.json")
    )

    if not files:
        raise FileNotFoundError(
            f"No problem_*.json files found inside "
            f"{processed_dir}"
        )

    required_fields = {
        "id",
        "source",
        "year",
        "topic",
        "subtopics",
        "statement",
        "solutions",
    }

    dataset = []

    for path in files:

        problem = load_json(path)

        missing = required_fields - problem.keys()

        if missing:
            raise ValueError(
                f"{path.name} is missing fields: "
                f"{sorted(missing)}"
            )

        problem["_source_file"] = path.name

        dataset.append(problem)

    ids = [
        str(problem["id"])
        for problem in dataset
    ]

    duplicates = [
        problem_id
        for problem_id, count in Counter(ids).items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate problem IDs found: {duplicates}"
        )

    return dataset


# ============================================================
# DETERMINE A1, A2, C1, G4, N6, ETC.
# ============================================================

def build_problem_index(
    dataset: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:

    index = {}

    groups = defaultdict(list)

    # If you later add problem_code directly to your JSON,
    # the script will automatically use it.

    for problem in dataset:

        year = int(problem["year"])

        explicit_code = problem.get("problem_code")

        if explicit_code:

            code = str(explicit_code).upper()

            key = (
                year,
                code,
            )

            if key in index:
                raise ValueError(
                    f"Duplicate code found: "
                    f"{year} {code}"
                )

            index[key] = problem

        else:

            topic = str(problem["topic"])

            if topic in TOPIC_TO_PREFIX:

                groups[
                    (
                        year,
                        topic,
                    )
                ].append(problem)

    # Infer the problem code using the ordering of IDs.
    #
    # Example:
    #
    # 2010 Algebra:
    # smallest ID = A1
    # next ID     = A2
    # next ID     = A3
    # etc.

    for (
        year,
        topic,
    ), problems in groups.items():

        problems.sort(key=numeric_id)

        prefix = TOPIC_TO_PREFIX[topic]

        for position, problem in enumerate(
            problems,
            start=1,
        ):

            code = f"{prefix}{position}"

            key = (
                year,
                code,
            )

            if key not in index:
                index[key] = problem

    return index


# ============================================================
# SELECT EXACTLY THE 64 BENCHMARK PROBLEMS
# ============================================================

def select_benchmark(
    index: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:

    selected = []

    missing = []

    subject_number = Counter()

    for spec in BENCHMARK_SPECS:

        key = (
            spec.year,
            spec.code,
        )

        problem = index.get(key)

        if problem is None:

            missing.append(
                f"{spec.year} {spec.code}"
            )

            continue

        prefix, _ = parse_problem_code(
            spec.code
        )

        expected_topic = PREFIX_TO_TOPIC[
            prefix
        ]

        if problem["topic"] != expected_topic:

            raise ValueError(
                f"{spec.year} {spec.code}: "
                f"expected topic {expected_topic}, "
                f"but found {problem['topic']}."
            )

        if not str(
            problem["statement"]
        ).strip():

            raise ValueError(
                f"{spec.year} {spec.code} "
                f"has an empty statement."
            )

        if not problem["solutions"]:

            raise ValueError(
                f"{spec.year} {spec.code} "
                f"has no reference solutions."
            )

        subject_number[
            expected_topic
        ] += 1

        benchmark_id = (
            f"IMO64-{prefix}"
            f"{subject_number[expected_topic]:02d}"
        )

        selected.append(
            {
                "benchmark_id": benchmark_id,

                "source_id": str(
                    problem["id"]
                ),

                "source_file": problem[
                    "_source_file"
                ],

                "source": problem[
                    "source"
                ],

                "year": str(
                    spec.year
                ),

                "problem_code": spec.code,

                "topic": expected_topic,

                "subtopics": problem.get(
                    "subtopics",
                    [],
                ),

                "difficulty_band": spec.level,

                "statement": problem[
                    "statement"
                ],

                "_solutions": problem[
                    "solutions"
                ],
            }
        )

    if missing:

        print()
        print("Missing benchmark problems:")

        for item in missing:
            print(f"  - {item}")

        raise ValueError(
            "\nSome selected IMO problems were not "
            "found in data/processed."
        )

    if len(selected) != 64:
        raise ValueError(
            f"Expected 64 problems, "
            f"selected {len(selected)}."
        )

    source_ids = [
        x["source_id"]
        for x in selected
    ]

    if len(source_ids) != len(
        set(source_ids)
    ):
        raise ValueError(
            "The same source problem was "
            "selected more than once."
        )

    return selected


# ============================================================
# MODEL-FACING QUESTION FORMAT
# ============================================================

def make_question_record(
    item: dict[str, Any],
) -> dict[str, Any]:

    # IMPORTANT:
    # NO SOLUTIONS ARE GIVEN TO THE MODEL.

    return {
        "benchmark_id": item[
            "benchmark_id"
        ],
        "source_id": item[
            "source_id"
        ],
        "source": item[
            "source"
        ],
        "year": item[
            "year"
        ],
        "problem_code": item[
            "problem_code"
        ],
        "topic": item[
            "topic"
        ],
        "subtopics": item[
            "subtopics"
        ],
        "difficulty_band": item[
            "difficulty_band"
        ],
        "statement": item[
            "statement"
        ],
    }


# ============================================================
# REFERENCE SOLUTION FORMAT
# ============================================================

def make_reference_record(
    item: dict[str, Any],
) -> dict[str, Any]:

    return {
        "benchmark_id": item[
            "benchmark_id"
        ],
        "source_id": item[
            "source_id"
        ],
        "year": item[
            "year"
        ],
        "problem_code": item[
            "problem_code"
        ],
        "topic": item[
            "topic"
        ],
        "difficulty_band": item[
            "difficulty_band"
        ],
        "solutions": item[
            "_solutions"
        ],
    }


# ============================================================
# WRITE BENCHMARK FILES
# ============================================================

def write_benchmark(
    selected: list[dict[str, Any]],
    output_dir: Path,
    include_references: bool,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = [
        make_question_record(item)
        for item in selected
    ]

    references = [
        make_reference_record(item)
        for item in selected
    ]

    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    write_jsonl(
        output_dir / "questions.jsonl",
        questions,
    )

    # --------------------------------------------------------
    # EXCLUSION LIST
    #
    # THIS IS VERY IMPORTANT.
    #
    # Future training code must NEVER train on
    # any problem listed here.
    # --------------------------------------------------------

    excluded = {
        "benchmark": "IMO64",

        "warning": (
            "These problems must never be used "
            "for training or fine-tuning."
        ),

        "source_ids": sorted(
            [
                x["source_id"]
                for x in selected
            ],
            key=int,
        ),

        "source_files": sorted(
            [
                x["source_file"]
                for x in selected
            ]
        ),

        "problems": [
            {
                "source_id": x[
                    "source_id"
                ],
                "year": x[
                    "year"
                ],
                "problem_code": x[
                    "problem_code"
                ],
                "topic": x[
                    "topic"
                ],
            }
            for x in selected
        ],
    }

    write_json(
        output_dir
        / "excluded_source_ids.json",
        excluded,
    )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    subject_counts = Counter(
        x["topic"]
        for x in selected
    )

    level_counts = Counter(
        x["difficulty_band"]
        for x in selected
    )

    manifest = {
        "name": "IMO64",

        "version": "1.0",

        "total_problems": 64,

        "description": (
            "Protected development benchmark "
            "for olympiad mathematical reasoning."
        ),

        "subjects": {
            topic: subject_counts[topic]
            for topic in TOPIC_TO_PREFIX
        },

        "difficulty_bands": {
            level: level_counts[level]
            for level in [
                "L1",
                "L2",
                "L3",
                "L4",
            ]
        },

        "pass_rule": {
            "overall_minimum": 52,
            "minimum_per_subject": 13,
            "subject_total": 16,
        },

        "important_rule": (
            "Benchmark problems and solutions "
            "must not appear in model training."
        ),

        "problems": [
            {
                "benchmark_id": x[
                    "benchmark_id"
                ],
                "source_id": x[
                    "source_id"
                ],
                "year": x[
                    "year"
                ],
                "problem_code": x[
                    "problem_code"
                ],
                "topic": x[
                    "topic"
                ],
                "difficulty_band": x[
                    "difficulty_band"
                ],
            }
            for x in selected
        ],
    }

    write_json(
        output_dir / "manifest.json",
        manifest,
    )

    # --------------------------------------------------------
    # OPTIONAL REFERENCE SOLUTIONS
    # --------------------------------------------------------

    reference_path = (
        output_dir
        / "references.jsonl"
    )

    if include_references:

        write_jsonl(
            reference_path,
            references,
        )

    elif reference_path.exists():

        reference_path.unlink()


# ============================================================
# PRINT RESULT
# ============================================================

def print_summary(
    selected: list[dict[str, Any]],
    output_dir: Path,
    include_references: bool,
) -> None:

    print()
    print("=" * 65)
    print("IMO64 BENCHMARK CREATED SUCCESSFULLY")
    print("=" * 65)

    print()
    print(
        f"Output folder: {output_dir}"
    )

    print()
    print("Problems by subject:")

    for topic in TOPIC_TO_PREFIX:

        count = sum(
            1
            for x in selected
            if x["topic"] == topic
        )

        print(
            f"  {topic}: {count}/16"
        )

    print()
    print("Problems by difficulty:")

    for level in [
        "L1",
        "L2",
        "L3",
        "L4",
    ]:

        count = sum(
            1
            for x in selected
            if x["difficulty_band"]
            == level
        )

        print(
            f"  {level}: {count}/16"
        )

    print()
    print("Files created:")

    print(
        "  benchmarks/imo64/"
        "questions.jsonl"
    )

    print(
        "  benchmarks/imo64/"
        "manifest.json"
    )

    print(
        "  benchmarks/imo64/"
        "excluded_source_ids.json"
    )

    if include_references:

        print(
            "  benchmarks/imo64/"
            "references.jsonl"
        )

    print()
    print("PASS CONDITION:")

    print(
        "  At least 52 / 64 overall"
    )

    print(
        "  AND at least 13 / 16 "
        "in every subject"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Never train on problems listed "
        "in excluded_source_ids.json"
    )

    print()
    print("=" * 65)


# ============================================================
# COMMAND LINE OPTIONS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build the protected IMO64 benchmark."
        )
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--include-references",
        action="store_true",
        help=(
            "Also create references.jsonl "
            "containing official solutions."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    args = parse_args()

    try:

        validate_benchmark_specs()

        dataset = load_processed_dataset(
            args.processed_dir
        )

        print(
            f"Loaded {len(dataset)} "
            f"processed problems."
        )

        index = build_problem_index(
            dataset
        )

        selected = select_benchmark(
            index
        )

        write_benchmark(
            selected,
            args.output_dir,
            args.include_references,
        )

        print_summary(
            selected,
            args.output_dir,
            args.include_references,
        )

        return 0

    except Exception as error:

        print()
        print("=" * 65)
        print("ERROR BUILDING IMO64")
        print("=" * 65)
        print()
        print(error)
        print()

        return 1


if __name__ == "__main__":
    sys.exit(main())