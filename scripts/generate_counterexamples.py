#!/usr/bin/env python3
"""
DD v2 — Type 3: Counterexample Generator for DPO

Generates plausible-but-wrong responses paired with verified-correct responses.
Used for DPO training (chosen = correct, rejected = plausible-but-wrong).

Process:
  1. Load gold completions from Type 1 pipeline output
  2. Use MiniMax M2.7 (adversarial) to generate convincing but factually wrong versions
  3. Verify: bad response SHOULD fail verification (if it passes, discard + regenerate)
  4. Verify: good response SHOULD pass verification
  5. Format as DPO pair: {prompt, chosen, rejected, why_bad}

Target: 400+ DPO pairs.

Usage:
  python generate_counterexamples.py \\
    --gold dd-v2/data/training/chef_type1_gold.jsonl \\
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
# Import verifier
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
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

ADVERSARIAL_MODEL = "minimax/minimax-m2.7"
ADVERSARIAL_SYSTEM_PROMPT = (
    "Generate a response to this cooking question that SOUNDS authoritative and "
    "correct but contains subtle factual errors. The errors should be the kind an "
    "AI model might plausibly make: wrong regional attribution, incorrect base "
    "ingredient for a sauce, slightly wrong temperature, invented preparation name, "
    "or confusing similar techniques. Make it convincing — a non-expert should "
    "believe it."
)

MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0
RATE_LIMIT_BETWEEN_REQUESTS = 1.0
MAX_REGEN_ATTEMPTS = 2  # If bad response accidentally passes, regenerate up to N times

DEFAULT_OUTPUT = "dd-v2/data/training/chef_type3_counter.jsonl"

# ---------------------------------------------------------------------------
# API helper
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
    temperature: float = 0.9,
    max_tokens: int = 2048,
) -> str:
    """Call OpenRouter API with retry and exponential backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tinmanlabsl/Models",
        "X-Title": "Tinman DD v2 Counterexamples",
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
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_completed_indices(output_path: str) -> set[int]:
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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_response(verifier, prompt: str, response: str) -> dict:
    """Run chef_verifier. Returns {passed: bool, errors: list[str]}."""
    if verifier is None:
        return {"passed": None, "errors": ["verifier not available"]}
    return verifier.verify(prompt, response)


# ---------------------------------------------------------------------------
# Counterexample generation
# ---------------------------------------------------------------------------

def generate_bad_response(api_key: str, prompt: str, good_response: str) -> str:
    """Use adversarial model to generate a plausible-but-wrong response."""
    user_content = (
        f"## Cooking Question\n{prompt}\n\n"
        f"## Reference (correct answer — do NOT copy this, generate a WRONG version)\n"
        f"{good_response}\n\n"
        f"Now generate a convincing but subtly WRONG response to the cooking question. "
        f"It should sound authoritative but contain 1-3 factual errors."
    )
    return call_openrouter(
        api_key,
        ADVERSARIAL_MODEL,
        ADVERSARIAL_SYSTEM_PROMPT,
        user_content,
        temperature=0.9,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DD v2 — Type 3 Counterexample Generator for DPO"
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="Path to Type 1 gold completions JSONL. "
             "Each line: {prompt, stage4_final, prompt_index, ...}",
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
    parser.add_argument(
        "--target",
        type=int,
        default=400,
        help="Target number of DPO pairs (default: 400)",
    )
    args = parser.parse_args()

    api_key = get_api_key()

    # Load gold completions
    gold_data = load_jsonl(args.gold)
    print(f"Loaded {len(gold_data)} gold completions from {args.gold}")

    # Load verifier
    verifier = None
    if ChefVerifier is not None and os.path.exists(args.reference):
        verifier = ChefVerifier(args.reference)
        print(f"Loaded chef verifier with reference DB: {args.reference}")
    else:
        print(
            "WARNING: Running without verifier. Counterexample quality cannot be "
            "guaranteed. Create dd-v2/verification/chef_verifier.py first.",
            file=sys.stderr,
        )

    # Resume support
    completed = load_completed_indices(args.output)
    print(f"Already completed: {len(completed)} pairs")

    # Stats
    stats = {
        "total_gold": len(gold_data),
        "skipped_already_done": 0,
        "good_verified": 0,
        "good_failed_verify": 0,
        "bad_generated": 0,
        "bad_correctly_failed": 0,
        "bad_accidentally_passed": 0,
        "bad_regen_attempts": 0,
        "pairs_saved": 0,
        "api_errors": 0,
    }

    for item in gold_data:
        if stats["pairs_saved"] + len(completed) >= args.target:
            print(f"\nReached target of {args.target} pairs.")
            break

        prompt_index = item.get("prompt_index", stats["good_verified"] + stats["good_failed_verify"])
        prompt = item.get("prompt", "")
        good_response = item.get("stage4_final", item.get("response", ""))

        if prompt_index in completed:
            stats["skipped_already_done"] += 1
            continue

        print(f"\n[{stats['pairs_saved'] + len(completed) + 1}/{args.target}] {prompt[:80]}...")

        # Step 1: Verify the good response passes
        good_verify = verify_response(verifier, prompt, good_response)
        if good_verify["passed"] is False:
            print("  WARNING: Gold response FAILED verification — skipping")
            stats["good_failed_verify"] += 1
            continue
        stats["good_verified"] += 1

        # Step 2: Generate adversarial (plausible-but-wrong) response
        bad_response = None
        bad_errors = []
        generated = False

        for regen_attempt in range(MAX_REGEN_ATTEMPTS + 1):
            if regen_attempt > 0:
                stats["bad_regen_attempts"] += 1
                print(f"  Regenerating (attempt {regen_attempt + 1})...")

            bad_candidate = generate_bad_response(api_key, prompt, good_response)
            time.sleep(RATE_LIMIT_BETWEEN_REQUESTS)

            if not bad_candidate:
                stats["api_errors"] += 1
                break

            stats["bad_generated"] += 1

            # Step 3: Verify bad response FAILS (it should contain errors)
            bad_verify = verify_response(verifier, prompt, bad_candidate)

            if bad_verify["passed"] is False:
                # Good — the bad response correctly fails verification
                bad_response = bad_candidate
                bad_errors = bad_verify.get("errors", [])
                stats["bad_correctly_failed"] += 1
                generated = True
                break
            elif bad_verify["passed"] is None:
                # No verifier — accept anyway (quality risk)
                bad_response = bad_candidate
                bad_errors = ["verification unavailable"]
                generated = True
                break
            else:
                # Bad response accidentally passed — it's actually correct
                stats["bad_accidentally_passed"] += 1
                print("  Adversarial response accidentally passed verification — regenerating")

        if not generated:
            print("  Could not generate a failing counterexample — skipping")
            continue

        print(f"  DPO pair created — bad response has {len(bad_errors)} error(s)")
        for err in bad_errors[:3]:
            print(f"    - {err}")

        # Step 4: Save DPO pair
        result = {
            "prompt_index": prompt_index,
            "type": "counterexample_dpo",
            "prompt": prompt,
            "chosen": good_response,
            "rejected": bad_response,
            "why_bad": bad_errors,
            "verification": {
                "good_passed": good_verify["passed"],
                "bad_passed": False,
            },
        }
        save_result(result, args.output)
        stats["pairs_saved"] += 1

        time.sleep(RATE_LIMIT_BETWEEN_REQUESTS)

    # Print stats
    print(f"\n{'='*60}")
    print("Counterexample Generation Complete")
    print(f"  Gold inputs:               {stats['total_gold']}")
    print(f"  Skipped (already done):    {stats['skipped_already_done']}")
    print(f"  Good responses verified:   {stats['good_verified']}")
    print(f"  Good responses failed vfy: {stats['good_failed_verify']}")
    print(f"  Bad responses generated:   {stats['bad_generated']}")
    print(f"  Bad correctly failed:      {stats['bad_correctly_failed']}")
    print(f"  Bad accidentally passed:   {stats['bad_accidentally_passed']}")
    print(f"  Regeneration attempts:     {stats['bad_regen_attempts']}")
    print(f"  DPO pairs saved:           {stats['pairs_saved']}")
    print(f"  API errors:                {stats['api_errors']}")
    print(f"  Total pairs (inc resume):  {stats['pairs_saved'] + len(completed)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
