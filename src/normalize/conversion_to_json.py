import json
from pathlib import Path
import re

RAW_DIR = Path("data/raw/manual")
PROCESSED_DIR = Path("data/processed")


def parse_problem(text):
    sections = re.split(
        r"(?m)^(ID|SOURCE|YEAR|TOPIC|SUBTOPICS|STATEMENT|SOLUTION \d+):\s*",
        text
    )

    problem = {
        "id": "",
        "source": "",
        "year": "",
        "topic": "",
        "subtopics": [],
        "statement": "",
        "solutions": []
    }

    for i in range(1, len(sections), 2):
        heading = sections[i]
        content = sections[i + 1].strip()

        if heading == "ID":
            problem["id"] = content

        elif heading == "SOURCE":
            problem["source"] = content

        elif heading == "YEAR":
            problem["year"] = content

        elif heading == "TOPIC":
            problem["topic"] = content

        elif heading == "SUBTOPICS":
            if content:
                problem["subtopics"] = [
                    x.strip() for x in content.split(",")
                ]

        elif heading == "STATEMENT":
            problem["statement"] = content

        elif heading.startswith("SOLUTION"):
            problem["solutions"].append(content)

    return problem


PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

for raw_file in RAW_DIR.glob("problem_*.txt"):
    text = raw_file.read_text(encoding="utf-8")

    problem = parse_problem(text)

    output_file = PROCESSED_DIR / f"{raw_file.stem}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(problem, f, indent=2, ensure_ascii=False)

    print(f"Created: {output_file}")