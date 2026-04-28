"""
Format Validator — validate training data examples for the French Chef
specialist model across four data types:

  Type 1: Gold completion (prompt + ideal response)
  Type 2: Failure trace (multi-turn error correction)
  Type 3: Counterexample (bad vs good response with explanation)
  Type 4: Boundary (deferral / qualification)

Usage:
    from format_validator import validate_any
    is_valid, issues = validate_any(example_dict)
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_RESPONSE_LEN = 100
MAX_RESPONSE_LEN = 3000

DEFERRAL_PHRASES = [
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "i do not know",
    "outside my expertise",
    "beyond my knowledge",
    "not certain",
    "cannot confirm",
    "i would need to verify",
    "i'd recommend consulting",
    "i would recommend consulting",
    "consult a",
    "please verify",
    "not qualified",
    "unable to confirm",
    "i may be wrong",
    "take this with",
    "grain of salt",
    "not entirely sure",
    "would need to check",
    "defer to",
    "best to consult",
    "honestly not sure",
    "that said, i'm uncertain",
    "might not be accurate",
    "i could be mistaken",
]


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def validate_gold_format(example: dict) -> tuple[bool, list[str]]:
    """Validate a Type 1 gold completion example.

    Expected keys: prompt, response, type="gold".
    Response must be non-empty and within reasonable length.
    """
    issues = []

    # Required keys
    if "type" not in example:
        issues.append("Missing 'type' field")
    elif example["type"] != "gold":
        issues.append(f"Expected type='gold', got type='{example['type']}'")

    if "prompt" not in example:
        issues.append("Missing 'prompt' field")
    elif not isinstance(example["prompt"], str) or not example["prompt"].strip():
        issues.append("'prompt' must be a non-empty string")

    if "response" not in example:
        issues.append("Missing 'response' field")
    elif not isinstance(example["response"], str) or not example["response"].strip():
        issues.append("'response' must be a non-empty string")
    else:
        resp_len = len(example["response"].strip())
        if resp_len < MIN_RESPONSE_LEN:
            issues.append(f"Response too short ({resp_len} chars, min {MIN_RESPONSE_LEN})")
        if resp_len > MAX_RESPONSE_LEN:
            issues.append(f"Response too long ({resp_len} chars, max {MAX_RESPONSE_LEN})")

    return (len(issues) == 0, issues)


def validate_failure_trace_format(example: dict) -> tuple[bool, list[str]]:
    """Validate a Type 2 failure trace example.

    Expected keys: type="failure_trace", turns (list of 4 dicts).
    Turns: [user_prompt, assistant_fail, user_error_feedback, assistant_fix].
    Each turn has 'role' and 'content'.
    """
    issues = []

    if "type" not in example:
        issues.append("Missing 'type' field")
    elif example["type"] != "failure_trace":
        issues.append(f"Expected type='failure_trace', got type='{example['type']}'")

    if "turns" not in example:
        issues.append("Missing 'turns' field")
        return (False, issues)

    turns = example["turns"]
    if not isinstance(turns, list):
        issues.append("'turns' must be a list")
        return (False, issues)

    if len(turns) != 4:
        issues.append(f"Expected exactly 4 turns, got {len(turns)}")
        # Continue checking what we have

    expected_roles = ["user", "assistant", "user", "assistant"]
    turn_labels = [
        "user prompt",
        "assistant fail",
        "user error feedback",
        "assistant fix",
    ]

    for i, (expected_role, label) in enumerate(zip(expected_roles, turn_labels)):
        if i >= len(turns):
            issues.append(f"Missing turn {i + 1} ({label})")
            continue
        turn = turns[i]
        if not isinstance(turn, dict):
            issues.append(f"Turn {i + 1} ({label}) must be a dict")
            continue
        role = turn.get("role", "")
        if role != expected_role:
            issues.append(f"Turn {i + 1} ({label}): expected role='{expected_role}', got '{role}'")
        content = turn.get("content", "")
        if not isinstance(content, str) or not content.strip():
            issues.append(f"Turn {i + 1} ({label}): content must be a non-empty string")

    return (len(issues) == 0, issues)


def validate_counterexample_format(example: dict) -> tuple[bool, list[str]]:
    """Validate a Type 3 counterexample.

    Expected keys: prompt, bad_response, why_bad, good_response, type="counterexample".
    """
    issues = []

    if "type" not in example:
        issues.append("Missing 'type' field")
    elif example["type"] != "counterexample":
        issues.append(f"Expected type='counterexample', got type='{example['type']}'")

    for field in ["prompt", "bad_response", "good_response"]:
        if field not in example:
            issues.append(f"Missing '{field}' field")
        elif not isinstance(example[field], str) or not example[field].strip():
            issues.append(f"'{field}' must be a non-empty string")

    if "why_bad" not in example:
        issues.append("Missing 'why_bad' field")
    elif not isinstance(example["why_bad"], str) or not example["why_bad"].strip():
        issues.append("'why_bad' must be a non-empty string explaining the error")
    elif len(example["why_bad"].strip()) < 10:
        issues.append("'why_bad' is too short to be a meaningful explanation")

    return (len(issues) == 0, issues)


def validate_boundary_format(example: dict) -> tuple[bool, list[str]]:
    """Validate a Type 4 boundary example.

    Expected keys: prompt, response, type="boundary".
    Response should contain deferral/qualification language.
    """
    issues = []

    if "type" not in example:
        issues.append("Missing 'type' field")
    elif example["type"] != "boundary":
        issues.append(f"Expected type='boundary', got type='{example['type']}'")

    if "prompt" not in example:
        issues.append("Missing 'prompt' field")
    elif not isinstance(example["prompt"], str) or not example["prompt"].strip():
        issues.append("'prompt' must be a non-empty string")

    if "response" not in example:
        issues.append("Missing 'response' field")
    elif not isinstance(example["response"], str) or not example["response"].strip():
        issues.append("'response' must be a non-empty string")
    else:
        resp_lower = example["response"].lower()
        has_deferral = any(phrase in resp_lower for phrase in DEFERRAL_PHRASES)
        if not has_deferral:
            issues.append(
                "Boundary response does not contain recognizable deferral/qualification language"
            )

    return (len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_VALIDATORS = {
    "gold": validate_gold_format,
    "failure_trace": validate_failure_trace_format,
    "counterexample": validate_counterexample_format,
    "boundary": validate_boundary_format,
}


def validate_any(example: dict) -> tuple[bool, list[str]]:
    """Route to the correct validator based on the 'type' field."""
    if not isinstance(example, dict):
        return (False, ["Example must be a dict"])

    example_type = example.get("type")
    if example_type is None:
        return (False, ["Missing 'type' field — cannot determine which validator to use"])

    validator = _VALIDATORS.get(example_type)
    if validator is None:
        valid_types = ", ".join(sorted(_VALIDATORS.keys()))
        return (False, [f"Unknown type '{example_type}'. Valid types: {valid_types}"])

    return validator(example)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python format_validator.py <json_file_or_jsonl_file>")
        print("  Validates each example in the file and reports issues.")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSON array first, then JSONL
    try:
        examples = json.loads(content)
        if isinstance(examples, dict):
            examples = [examples]
    except json.JSONDecodeError:
        examples = []
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Line {line_num}: invalid JSON, skipping")

    total = len(examples)
    passed = 0
    failed = 0

    for i, ex in enumerate(examples):
        is_valid, issues = validate_any(ex)
        if is_valid:
            passed += 1
        else:
            failed += 1
            ex_type = ex.get("type", "unknown")
            print(f"Example {i + 1} (type={ex_type}): FAIL")
            for issue in issues:
                print(f"  - {issue}")

    print(f"\n{passed}/{total} passed, {failed}/{total} failed")
