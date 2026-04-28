#!/usr/bin/env python3
"""
DD v2 — Type 2: Failure Trace Generator

Generates multi-turn failure-aware training examples:
  1. Load student (base model) responses
  2. Verify against French cuisine reference DB
  3. For failures: critique errors via multi-teacher pipeline, synthesize fix
  4. Re-verify the corrected response
  5. Keep only verified corrections
  6. Output format: [user prompt, student fail, error feedback, verified fix]

Usage:
  python generate_failure_traces.py \\
    --student-responses dd-v2/data/chef_student_responses.jsonl \\
    --prompts dd-v2/data/prompts/chef_prompts.jsonl \\
    --reference dd-v2/data/french_cuisine_reference.json

Requires: OPENROUTER_API_KEY env var
"""

import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Import verifier — lives in dd-v2/verification/
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "verification"))

try:
    from chef_verifier import ChefVerifier
except ImportError:
    print(
        "WARNING: chef_verifier not found in dd-v2/verification/. "
        "Create it before running this script.",
        file=sys.stderr,
    )
    ChefVerifier = None

# ---------------------------------------------------------------------------
# Constants (shared with pipeline_runner)
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

STAGE_MODELS = {
    1: "moonshotai/kimi-k2.5",
    2: "google/gemma-4-26b-a4b-it",
    3: "minimax/minimax-m2.7",
    4: "qwen/qwen3.6-plus:free",
}

RATE_LIMIT_BETWEEN_REQUESTS = 1.0
RATE_LIMIT_BETWEEN_STAGES = 3.0
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0

DEFAULT_OUTPUT = "dd-v2/data/training/chef_type2_failure.jsonl"

# ---------------------------------------------------------------------------
# API helper (same as pipeline_runner)
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    return key


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Call OpenRouter API with retry and exponential backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tinmanlabsl/Models",
        "X-Title": "Tinman DD v2 Failure Traces",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code in (429, 500, 502, 503):
                print(
                    f"  Retry {attempt}/{MAX_RETRIES} — HTTP {resp.status_code}, "
                    f"waiting {backoff:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                print(
                    f"  ERROR: HTTP {resp.status_code}: {resp.text[:300]}",
                    file=sys.stderr,
                )
                return ""
        except requests.exceptions.Timeout:
            print(
                f"  Retry {attempt}/{MAX_RETRIES} — timeout, waiting {backoff:.1f}s",
                file=sys.stderr,
            )
            time.sleep(backoff)
            backoff *= 2
        except requests.exceptions.RequestException as e:
            print(f"  ERROR: request failed: {e}", file=sys.stderr)
            return ""

    print("  ERROR: all retries exhausted.", file=sys.stderr)
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_completed_indices(output_path: str) -> set[int]:
    """Return set of prompt indices already in the output file."""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                completed.add(obj.get("prompt_index", -1))
            except json.JSONDecodeError:
                continue
    return completed


def save_result(result: dict, output_path: str) -> None:
    """Append a single result to the output JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_response(verifier, prompt: str, response: str) -> dict:
    """Run chef_verifier on a response. Returns {passed: bool, errors: list[str]}."""
    if verifier is None:
        # Fallback: no verifier available, cannot verify
        return {"passed": None, "errors": ["verifier not available"]}
    result = verifier.verify(prompt, response)
    return result


# ---------------------------------------------------------------------------
# Multi-teacher critique on specific errors
# ---------------------------------------------------------------------------

def format_error_feedback(errors: list[str]) -> str:
    """Format verification errors into human-readable feedback."""
    if not errors:
        return "No specific errors identified."
    feedback_lines = ["The following errors were found in your response:"]
    for i, err in enumerate(errors, 1):
        feedback_lines.append(f"  {i}. {err}")
    return "\n".join(feedback_lines)


def run_error_critique(
    api_key: str,
    prompt: str,
    student_response: str,
    errors: list[str],
) -> dict[int, str]:
    """Run stages 1-3 to critique the specific errors in the student response."""
    error_feedback = format_error_feedback(errors)
    critiques = {}

    # Stage 1 (Eager): Warm correction
    sys1 = (
        "You are an enthusiastic French cuisine expert helping a student correct "
        "mistakes. Be encouraging but precise about what went wrong and why."
    )
    user1 = (
        f"## Student's Question\n{prompt}\n\n"
        f"## Student's Response\n{student_response}\n\n"
        f"## Identified Errors\n{error_feedback}\n\n"
        f"Please explain what went wrong and provide the correct information."
    )
    critiques[1] = call_openrouter(api_key, STAGE_MODELS[1], sys1, user1)
    time.sleep(RATE_LIMIT_BETWEEN_STAGES)

    # Stage 2 (Deliberate): Precise correction with details
    sys2 = (
        "You are a meticulous French culinary scholar. Given a student's incorrect "
        "response and the specific errors identified, provide a thorough correction "
        "with precise details: correct techniques, temperatures, regional origins, "
        "and classical preparations."
    )
    user2 = (
        f"## Original Question\n{prompt}\n\n"
        f"## Student's Incorrect Response\n{student_response}\n\n"
        f"## Identified Errors\n{error_feedback}\n\n"
        f"## Stage 1 Correction Attempt\n{critiques[1]}\n\n"
        f"Provide a detailed, precise correction."
    )
    critiques[2] = call_openrouter(api_key, STAGE_MODELS[2], sys2, user2)
    time.sleep(RATE_LIMIT_BETWEEN_STAGES)

    # Stage 3 (Adversarial): Verify the corrections themselves
    sys3 = (
        "You are a critical French cuisine reviewer. Review the corrections below "
        "and check: are the corrections themselves accurate? Are there remaining "
        "errors or over-corrections? Be thorough."
    )
    user3 = (
        f"## Original Question\n{prompt}\n\n"
        f"## Student's Incorrect Response\n{student_response}\n\n"
        f"## Identified Errors\n{error_feedback}\n\n"
        f"## Stage 1 Correction\n{critiques[1]}\n\n"
        f"## Stage 2 Correction\n{critiques[2]}\n\n"
        f"Review these corrections for accuracy."
    )
    critiques[3] = call_openrouter(api_key, STAGE_MODELS[3], sys3, user3)

    return critiques


def synthesize_correction(
    api_key: str,
    prompt: str,
    student_response: str,
    errors: list[str],
    critiques: dict[int, str],
) -> str:
    """Stage 4: Synthesize a corrected response incorporating all critiques."""
    error_feedback = format_error_feedback(errors)

    sys4 = (
        "You are a French cuisine expert synthesizer. A student made errors in their "
        "response. Multiple experts have analyzed the errors and proposed corrections. "
        "Produce a final corrected response that is factually accurate and naturally "
        "written. Do NOT reference the error correction process — just give the "
        "correct answer as a knowledgeable chef would."
    )
    user4 = (
        f"## Original Question\n{prompt}\n\n"
        f"## Student's Incorrect Response\n{student_response}\n\n"
        f"## Identified Errors\n{error_feedback}\n\n"
        f"## Expert Correction 1 (Eager)\n{critiques.get(1, '')}\n\n"
        f"## Expert Correction 2 (Deliberate)\n{critiques.get(2, '')}\n\n"
        f"## Expert Review (Adversarial)\n{critiques.get(3, '')}\n\n"
        f"Produce the final corrected response."
    )

    return call_openrouter(api_key, STAGE_MODELS[4], sys4, user4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DD v2 — Type 2 Failure Trace Generator"
    )
    parser.add_argument(
        "--student-responses",
        required=True,
        help="Path to student (base model) responses JSONL. "
             "Each line: {prompt, response, prompt_index}",
    )
    parser.add_argument(
        "--prompts",
        default="dd-v2/data/prompts/chef_prompts.jsonl",
        help="Path to original prompts JSONL (for metadata)",
    )
    parser.add_argument(
        "--reference",
        default="dd-v2/data/french_cuisine_reference.json",
        help="Path to French cuisine reference DB JSON",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    api_key = get_api_key()

    # Load data
    student_data = load_jsonl(args.student_responses)
    print(f"Loaded {len(student_data)} student responses from {args.student_responses}")

    # Load verifier
    verifier = None
    if ChefVerifier is not None and os.path.exists(args.reference):
        verifier = ChefVerifier(args.reference)
        print(f"Loaded chef verifier with reference DB: {args.reference}")
    else:
        print(
            "WARNING: Running without verifier. Create dd-v2/verification/chef_verifier.py "
            "and dd-v2/data/french_cuisine_reference.json first.",
            file=sys.stderr,
        )

    # Resume support
    completed = load_completed_indices(args.output)
    print(f"Already completed: {len(completed)} prompts")

    # Stats
    stats = {
        "total": 0,
        "skipped_already_done": 0,
        "student_passed": 0,
        "student_failed": 0,
        "corrections_generated": 0,
        "corrections_verified": 0,
        "corrections_failed_reverify": 0,
        "api_errors": 0,
    }

    for item in student_data:
        prompt_index = item.get("prompt_index", stats["total"])
        prompt = item.get("prompt", "")
        student_response = item.get("response", "")
        stats["total"] += 1

        if prompt_index in completed:
            stats["skipped_already_done"] += 1
            continue

        print(f"\n[{stats['total']}/{len(student_data)}] {prompt[:80]}...")

        # Step 1: Verify student response
        vresult = verify_response(verifier, prompt, student_response)

        if vresult["passed"] is True:
            stats["student_passed"] += 1
            print("  Student response PASSED verification — skipping (no failure trace)")
            continue

        if vresult["passed"] is None:
            # No verifier — use all responses as potential failures
            # In production, this should not happen
            print("  WARNING: no verifier, treating as failure")

        stats["student_failed"] += 1
        errors = vresult.get("errors", [])
        print(f"  Student FAILED — {len(errors)} error(s) found")
        for err in errors[:3]:
            print(f"    - {err}")

        # Step 2: Multi-teacher critique on the specific errors
        print("  Running multi-teacher critique...")
        critiques = run_error_critique(api_key, prompt, student_response, errors)
        time.sleep(RATE_LIMIT_BETWEEN_STAGES)

        if not critiques.get(1) or not critiques.get(2) or not critiques.get(3):
            print("  ERROR: critique stages incomplete, skipping")
            stats["api_errors"] += 1
            continue

        # Step 3: Synthesize corrected response
        print("  Synthesizing correction...")
        correction = synthesize_correction(
            api_key, prompt, student_response, errors, critiques
        )
        time.sleep(RATE_LIMIT_BETWEEN_STAGES)

        if not correction:
            print("  ERROR: synthesis failed, skipping")
            stats["api_errors"] += 1
            continue

        stats["corrections_generated"] += 1

        # Step 4: Re-verify the correction
        reverify = verify_response(verifier, prompt, correction)
        if reverify["passed"] is False:
            print(f"  Correction FAILED re-verification — discarding")
            for err in reverify.get("errors", [])[:3]:
                print(f"    - {err}")
            stats["corrections_failed_reverify"] += 1
            continue

        stats["corrections_verified"] += 1
        print("  Correction PASSED re-verification")

        # Step 5: Format as multi-turn training example
        error_feedback = format_error_feedback(errors)
        result = {
            "prompt_index": prompt_index,
            "type": "failure_trace",
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": student_response},
                {"role": "user", "content": error_feedback},
                {"role": "assistant", "content": correction},
            ],
            "errors_found": errors,
            "verification": {
                "student_passed": False,
                "correction_passed": reverify["passed"],
            },
        }
        save_result(result, args.output)

        time.sleep(RATE_LIMIT_BETWEEN_REQUESTS)

    # Print stats
    print(f"\n{'='*60}")
    print("Failure Trace Generation Complete")
    print(f"  Total prompts:             {stats['total']}")
    print(f"  Skipped (already done):    {stats['skipped_already_done']}")
    print(f"  Student passed (no trace): {stats['student_passed']}")
    print(f"  Student failed:            {stats['student_failed']}")
    print(f"  Corrections generated:     {stats['corrections_generated']}")
    print(f"  Corrections verified:      {stats['corrections_verified']}")
    print(f"  Corrections failed re-vfy: {stats['corrections_failed_reverify']}")
    print(f"  API errors:                {stats['api_errors']}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
