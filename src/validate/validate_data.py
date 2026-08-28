import json
import re
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
seen_ids = set()
total_checked = 0
total_passed = 0
total_failed = 0

for json_file in PROCESSED_DIR.glob("problem_*.json"):
    total_checked += 1
    print(f"Checking: {json_file}")

    try:
        with json_file.open("r", encoding="utf-8") as f:
            problem = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON - {e}")
        total_failed += 1
        print()
        continue

    print(f"Loaded problem {problem.get('id', 'UNKNOWN')}")

    has_error = False

    required_fields = ["id", "statement", "source", "year", "topic"]

    if problem.get("id"):
        if problem["id"] in seen_ids:
            print(f"ERROR: Duplicate ID '{problem['id']}'")
            has_error = True
        else:
            seen_ids.add(problem["id"])

    for field in required_fields:
        if field not in problem:
            print(f"ERROR: Missing field '{field}'")
            has_error = True
        elif not problem[field]:
            print(f"ERROR: Field '{field}' is empty")
            has_error = True

    if problem.get("id") and not re.fullmatch(r"\d{4}", problem["id"]):
        print(f"ERROR: ID '{problem['id']}' must be exactly 4 digits")
        has_error = True

    expected_id = json_file.stem.replace("problem_", "")

    if problem.get("id") and problem["id"] != expected_id:
        print(
            f"ERROR: ID '{problem['id']}' does not match "
            f"filename ID '{expected_id}'"
        )
        has_error = True
    if not has_error:
        print(f"PASS: Problem {problem['id']} passed all checks.")
        total_passed += 1
    else:
        total_failed += 1

    print()
print("----- SUMMARY -----")
print(f"Checked: {total_checked}")
print(f"Passed:  {total_passed}")
print(f"Failed:  {total_failed}")