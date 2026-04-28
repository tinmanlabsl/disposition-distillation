#!/usr/bin/env python3
"""
DD v2 — Unified 4-Stage Multi-Teacher Pipeline Runner

Orchestrates the full teacher pipeline for French Chef training data generation.
Each prompt passes through 4 stages:
  Stage 1 (Eager):       Kimi K2.5 — warm, enthusiastic initial response
  Stage 2 (Deliberate):  Gemma 4 26B-A4B — careful, detailed analysis
  Stage 3 (Adversarial): MiniMax M2.7 — critique stages 1-2, find errors
  Stage 4 (Synthesizer): Qwen 3.6 Plus — final synthesis incorporating critique

All API calls via OpenRouter: https://openrouter.ai/api/v1/chat/completions
API key from env var OPENROUTER_API_KEY.

Usage:
  python pipeline_runner.py --prompts chef_prompts.jsonl
  python pipeline_runner.py --prompts chef_prompts.jsonl --start 100 --end 200
  python pipeline_runner.py --prompts chef_prompts.jsonl --stage 3  # debug single stage
"""

import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

STAGE_MODELS = {
    1: "moonshotai/kimi-k2.5",
    2: "google/gemma-4-26b-a4b-it",
    3: "minimax/minimax-m2.7",
    4: "qwen/qwen3.6-plus:free",
}

STAGE_SYSTEM_PROMPTS = {
    1: (
        "You are an enthusiastic French cuisine expert. Respond with warmth and "
        "passion about French cooking. Be detailed and accurate about techniques, "
        "ingredients, and regional traditions."
    ),
    2: (
        "You are a meticulous French culinary scholar. Analyze the following cooking "
        "question and the initial response carefully. Provide a thorough, precise "
        "answer with specific details about techniques, temperatures, regional "
        "origins, and classical preparations. Correct any inaccuracies in the "
        "initial response."
    ),
    3: (
        "You are a critical French cuisine reviewer. Your job is to find errors, "
        "omissions, and inaccuracies in culinary advice. Examine the responses below "
        "and identify: factual errors, missing important details, incorrect regional "
        "attributions, wrong techniques or temperatures, and any overconfident claims "
        "about debatable topics."
    ),
    4: (
        "You are a French cuisine expert synthesizer. Given a cooking question, "
        "multiple expert responses, and a critical review, produce a final "
        "authoritative answer. Incorporate valid corrections from the critique. "
        "Where facts are uncertain, say so. Be precise about techniques, ingredients, "
        "regions, and temperatures. Do NOT use markdown headers or bullet points "
        "excessively — write naturally as a knowledgeable chef would explain to a "
        "student."
    ),
}

RATE_LIMIT_BETWEEN_REQUESTS = 1.0   # seconds
RATE_LIMIT_BETWEEN_STAGES = 3.0     # seconds
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0               # seconds, doubles each retry

DEFAULT_OUTPUT = "dd-v2/data/training/chef_pipeline_raw.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    return key


def load_prompts(path: str) -> list[dict]:
    """Load prompts from a JSONL file.  Each line must have at least a 'prompt' field."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed line {i+1}: {e}", file=sys.stderr)
                continue
            if "prompt" not in obj:
                print(f"WARNING: line {i+1} has no 'prompt' field, skipping.", file=sys.stderr)
                continue
            prompts.append(obj)
    print(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def load_completed(output_path: str) -> set[int]:
    """Return set of prompt indices already completed in the output file."""
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
        "X-Title": "Tinman DD v2 Pipeline",
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
# Stage runners
# ---------------------------------------------------------------------------

def build_stage_user_content(
    stage: int, prompt: str, previous: dict[int, str]
) -> str:
    """Build the user-facing content for each stage."""
    if stage == 1:
        return prompt

    if stage == 2:
        return (
            f"## Original Question\n{prompt}\n\n"
            f"## Initial Response (Stage 1 — Eager)\n{previous[1]}"
        )

    if stage == 3:
        return (
            f"## Original Question\n{prompt}\n\n"
            f"## Stage 1 Response (Eager)\n{previous[1]}\n\n"
            f"## Stage 2 Response (Deliberate)\n{previous[2]}"
        )

    if stage == 4:
        return (
            f"## Original Question\n{prompt}\n\n"
            f"## Stage 1 Response (Eager)\n{previous[1]}\n\n"
            f"## Stage 2 Response (Deliberate)\n{previous[2]}\n\n"
            f"## Stage 3 Critique (Adversarial)\n{previous[3]}"
        )

    raise ValueError(f"Unknown stage: {stage}")


def run_stage(
    api_key: str,
    stage: int,
    prompt: str,
    previous: dict[int, str],
) -> str:
    """Run a single pipeline stage and return the response text."""
    model = STAGE_MODELS[stage]
    system_prompt = STAGE_SYSTEM_PROMPTS[stage]
    user_content = build_stage_user_content(stage, prompt, previous)

    print(f"    Stage {stage} ({model.split('/')[-1]})...", end=" ", flush=True)
    response = call_openrouter(api_key, model, system_prompt, user_content)
    if response:
        print(f"OK ({len(response)} chars)")
    else:
        print("FAILED")
    return response


def run_full_pipeline(
    api_key: str,
    prompt: str,
    single_stage: int | None = None,
) -> dict:
    """Run all 4 stages (or a single stage) and return results dict."""
    previous: dict[int, str] = {}
    stages = [single_stage] if single_stage else [1, 2, 3, 4]

    for stage in stages:
        # If running a single late stage, we need earlier stages' outputs.
        # In single-stage debug mode, fill missing previous stages with placeholders.
        for dep in range(1, stage):
            if dep not in previous:
                previous[dep] = "[not available — single-stage debug mode]"

        response = run_stage(api_key, stage, prompt, previous)
        if not response:
            return {"error": f"Stage {stage} failed", "partial": previous}
        previous[stage] = response

        # Rate limit between stages
        if stage != stages[-1]:
            time.sleep(RATE_LIMIT_BETWEEN_STAGES)

    result = {f"stage{s}": previous.get(s, "") for s in [1, 2, 3, 4]}
    result["stage4_final"] = previous.get(stages[-1], "")
    return result


def save_result(result: dict, output_path: str) -> None:
    """Append a single result to the output JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DD v2 — 4-Stage Multi-Teacher Pipeline Runner"
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to prompts JSONL file (each line has a 'prompt' field)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start at prompt index N (0-based)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End at prompt index N (exclusive)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Run only a single stage (for debugging)",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    prompts = load_prompts(args.prompts)

    # Slice
    end = args.end if args.end is not None else len(prompts)
    prompts_slice = prompts[args.start : end]
    print(f"Processing prompts [{args.start}:{end}] ({len(prompts_slice)} total)")

    # Resume support
    completed = load_completed(args.output)
    skipped = 0

    total = len(prompts_slice)
    succeeded = 0
    failed = 0

    for i, prompt_obj in enumerate(prompts_slice):
        global_idx = args.start + i
        if global_idx in completed:
            skipped += 1
            continue

        prompt_text = prompt_obj["prompt"]
        print(f"\n[{global_idx+1}/{end}] {prompt_text[:80]}...")

        result = run_full_pipeline(api_key, prompt_text, single_stage=args.stage)

        if "error" in result:
            print(f"  PIPELINE FAILED: {result['error']}")
            failed += 1
            # Still save partial result so we can inspect
            result["prompt"] = prompt_text
            result["prompt_index"] = global_idx
            result["metadata"] = prompt_obj.get("metadata", {})
            result["status"] = "failed"
            save_result(result, args.output)
        else:
            result["prompt"] = prompt_text
            result["prompt_index"] = global_idx
            result["metadata"] = prompt_obj.get("metadata", {})
            result["status"] = "complete"
            save_result(result, args.output)
            succeeded += 1

        # Rate limit between prompts
        time.sleep(RATE_LIMIT_BETWEEN_REQUESTS)

    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  Processed: {succeeded + failed}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Output:    {args.output}")


if __name__ == "__main__":
    main()
