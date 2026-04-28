#!/usr/bin/env python3
"""DD v2 Step 11b — Generate Type 2 Failure Traces

Compares student (Gemma 4 E2B base) responses against gold (4-stage teacher)
responses. For items where the student has factual errors, missing content,
or significantly lower quality:

1. Run ChefVerifier on both student and gold to quantify the gap
2. For failures: use DeepSeek V3.2 to generate a specific, targeted critique
   of the student's errors (NOT a rewrite — just the errors)
3. Use Qwen 3.6 Plus to synthesize the corrected response
4. Re-verify the corrected response

Output: multi-turn failure traces for SFT training
Format: prompt → student_attempt → error_feedback → corrected_response

Uses per-item caching and resume support.
"""
import asyncio
import json
import logging
import os
import re
import signal
import sys
import time

signal.signal(signal.SIGHUP, signal.SIG_IGN)

# ── paths ──────────────────────────────────────────────────────────────────
PERSIST = "/workspace/persistent"
STUDENT_PATH = os.path.join(PERSIST, "data", "student_responses.jsonl")
GOLD_PATH = os.path.join(PERSIST, "data", "type1_gold.jsonl")
OUTPUT_PATH = os.path.join(PERSIST, "data", "type2_failure.jsonl")
CACHE_DIR = os.path.join(PERSIST, "cache", "step11b")
LOG_PATH = os.path.join(PERSIST, "logs", "step11b_failure.log")
REF_DB_PATH = os.path.join(PERSIST, "dd-v2", "data", "french_cuisine_reference.json")
VERIFIER_DIR = os.path.join(PERSIST, "dd-v2", "verification")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── API config ─────────────────────────────────────────────────────────────
# DeepSeek V3.2 for error critique (precise, structured)
DEEPSEEK_MODEL = "deepseek/deepseek-v3.2"
DEEPSEEK_PROVIDER = {"order": ["novita"], "quantizations": ["fp8"]}
DEEPSEEK_TEMPERATURE = 0.6  # lower temp for precise error identification

# Qwen 3.6 Plus for corrected synthesis
QWEN_MODEL = "qwen/qwen3.6-plus:free"
QWEN_PROVIDER = {"order": ["alibaba"]}
QWEN_TEMPERATURE = 0.7

OR_CONCURRENCY = 8
QWEN_CONCURRENCY = 1
BATCH_SIZE = 20
MAX_RETRIES = 5
RETRY_DELAY = 8

# ── Failure detection thresholds ───────────────────────────────────────────
# Student is a "failure" if:
#   score_gap >= 0.15 (gold scores 15+ pts higher than student)
#   OR student has any incorrect/implausible/invented claims
#   OR student_score < 0.35 (absolute low bar)
#   OR gold_response is significantly longer (student omitted key content)
SCORE_GAP_THRESHOLD = 0.15
ABSOLUTE_LOW_THRESHOLD = 0.35
LENGTH_RATIO_THRESHOLD = 0.3  # student < 30% of gold length = likely omitted

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── shutdown ───────────────────────────────────────────────────────────────
_shutdown = False
def _handle_term(sig, frame):
    global _shutdown
    _shutdown = True
    log.warning("Shutdown requested (signal %d). Finishing current batch...", sig)

signal.signal(signal.SIGTERM, _handle_term)
signal.signal(signal.SIGINT, _handle_term)

# ── verifier import ────────────────────────────────────────────────────────
sys.path.insert(0, VERIFIER_DIR)
from chef_verifier import ChefVerifier


# ── prompts ────────────────────────────────────────────────────────────────

CRITIQUE_PROMPT = """You are a meticulous culinary fact-checker. A student (AI model) attempted to answer a French cuisine question. Compare their response against a verified reference answer and identify SPECIFIC factual errors, omissions, or misleading claims.

QUESTION:
{prompt}

STUDENT'S RESPONSE:
{student_response}

REFERENCE ANSWER (verified correct):
{gold_response}

AUTOMATED VERIFICATION FOUND THESE ISSUES:
{verification_issues}

Your task:
1. List each specific factual error in the student's response (with the correct information from the reference)
2. Note any critical omissions — important techniques, safety warnings, or key facts the student missed
3. Flag any vague or misleading claims that could confuse a cooking student

Format your response as a numbered list of specific, actionable corrections. Be precise — cite exact quotes from the student's response and the correct information. Do NOT rewrite the response, just identify errors.

If the student's response is actually correct but just less detailed, say "No factual errors found — response is correct but incomplete" and list what was omitted."""

CORRECTION_PROMPT = """You are a master French cuisine instructor. A student gave a response to a culinary question, and an expert reviewer identified specific errors. Write a CORRECTED version that:

1. Fixes all identified errors with correct information
2. Fills in critical omissions
3. Keeps the student's natural voice and structure where possible
4. Does NOT add unnecessary detail — just fix what's wrong

QUESTION:
{prompt}

STUDENT'S ORIGINAL RESPONSE:
{student_response}

ERRORS IDENTIFIED:
{critique}

Write the corrected response. Keep it concise and natural — this should read like a knowledgeable chef explaining, not a textbook. Use /no_think mode (no reasoning tags)."""


# ── cache helpers ──────────────────────────────────────────────────────────

def cache_path(item_id: str, stage: str) -> str:
    return os.path.join(CACHE_DIR, f"{item_id}_{stage}.json")


def load_cache(item_id: str, stage: str):
    p = cache_path(item_id, stage)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(item_id: str, stage: str, data: dict):
    p = cache_path(item_id, stage)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ── API call helper ────────────────────────────────────────────────────────

async def call_api(client, model: str, prompt: str, sem: asyncio.Semaphore,
                   label: str, temperature: float = 0.7, max_tokens: int = 4096,
                   extra_body: dict = None):
    """Call OpenRouter API with retries and rate-limit handling."""
    for attempt in range(MAX_RETRIES):
        if _shutdown:
            return None
        try:
            async with sem:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                resp = await client.chat.completions.create(**kwargs)

            msg = resp.choices[0].message
            text = msg.content or ""

            # Strip MiniMax-style think tags if present
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            if not text:
                log.warning(f"[{label}] Empty response, attempt {attempt+1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue

            return text

        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                wait = RETRY_DELAY * (attempt + 1)
                log.warning(f"[{label}] 429 rate limit, waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
                continue
            if "402" in err_str or "credit" in err_str.lower():
                log.error(f"[{label}] Credit exhausted: {e}")
                return None
            log.warning(f"[{label}] Attempt {attempt+1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)

    log.error(f"[{label}] All retries failed")
    return None


# ── failure detection ──────────────────────────────────────────────────────

def detect_failure(student_response: str, gold_response: str,
                   prompt: str, verifier: ChefVerifier) -> dict:
    """Determine if student response is a 'failure' worth creating a trace for."""
    student_score = verifier.score_response(student_response, prompt)
    gold_score = verifier.score_response(gold_response, prompt)
    student_verify = verifier.verify_response(student_response, prompt)
    gold_verify = verifier.verify_response(gold_response, prompt)

    reasons = []

    # Check score gap
    score_gap = gold_score - student_score
    if score_gap >= SCORE_GAP_THRESHOLD:
        reasons.append(f"score_gap={score_gap:.2f} (student={student_score:.2f}, gold={gold_score:.2f})")

    # Check absolute low score
    if student_score < ABSOLUTE_LOW_THRESHOLD:
        reasons.append(f"low_score={student_score:.2f}")

    # Check for specific errors in student response
    issues = student_verify.get("issues", [])
    if issues:
        reasons.append(f"verification_issues: {'; '.join(issues)}")

    # Check length ratio (student much shorter = likely omitted key content)
    if len(gold_response) > 200:
        ratio = len(student_response) / len(gold_response)
        if ratio < LENGTH_RATIO_THRESHOLD:
            reasons.append(f"length_ratio={ratio:.2f} (student={len(student_response)}, gold={len(gold_response)})")

    is_failure = len(reasons) > 0

    return {
        "is_failure": is_failure,
        "reasons": reasons,
        "student_score": student_score,
        "gold_score": gold_score,
        "score_gap": score_gap,
        "student_issues": issues,
        "student_length": len(student_response),
        "gold_length": len(gold_response),
    }


# ── process single failure ─────────────────────────────────────────────────

async def process_failure(item: dict, or_client, or_sem, qwen_sem, verifier):
    """Generate critique + corrected response for one failure."""
    item_id = item["id"]

    # Check cache for completed trace
    cached = load_cache(item_id, "trace")
    if cached:
        return cached

    prompt = item["prompt"]
    student_response = item["student_response"]
    gold_response = item["gold_response"]
    failure_info = item["failure_info"]

    # Step 1: DeepSeek critique of specific errors
    cached_critique = load_cache(item_id, "critique")
    if cached_critique:
        critique = cached_critique["text"]
    else:
        verification_issues = "\n".join(f"- {i}" for i in failure_info.get("student_issues", []))
        if not verification_issues:
            verification_issues = "\n".join(f"- {r}" for r in failure_info.get("reasons", []))

        critique_prompt = CRITIQUE_PROMPT.format(
            prompt=prompt,
            student_response=student_response,
            gold_response=gold_response,
            verification_issues=verification_issues or "No specific issues flagged by automated checks (but score gap or length gap detected).",
        )

        critique = await call_api(
            or_client, DEEPSEEK_MODEL, critique_prompt, or_sem,
            label=f"critique-{item_id}",
            temperature=DEEPSEEK_TEMPERATURE,
            max_tokens=2048,
            extra_body={"provider": DEEPSEEK_PROVIDER},
        )

        if not critique:
            log.warning(f"[{item_id}] Critique failed, skipping")
            return None

        save_cache(item_id, "critique", {"text": critique})

    # Step 2: Qwen synthesize corrected response
    cached_correction = load_cache(item_id, "correction")
    if cached_correction:
        corrected = cached_correction["text"]
    else:
        correction_prompt = CORRECTION_PROMPT.format(
            prompt=prompt,
            student_response=student_response,
            critique=critique,
        )

        corrected = await call_api(
            or_client, QWEN_MODEL, correction_prompt, qwen_sem,
            label=f"correct-{item_id}",
            temperature=QWEN_TEMPERATURE,
            max_tokens=4096,
            extra_body={"provider": QWEN_PROVIDER},
        )

        if not corrected:
            log.warning(f"[{item_id}] Correction failed, skipping")
            return None

        save_cache(item_id, "correction", {"text": corrected})

    # Step 3: Verify corrected response
    corrected_score = verifier.score_response(corrected, prompt)
    corrected_verify = verifier.verify_response(corrected, prompt)

    # Build the failure trace
    trace = {
        "id": item_id,
        "prompt": prompt,
        "category": item.get("category", ""),
        "difficulty": item.get("difficulty", ""),
        "type": "type2_failure_trace",
        # Multi-turn training format
        "student_attempt": student_response,
        "error_feedback": critique,
        "corrected_response": corrected,
        # Metadata for quality filtering
        "student_score": failure_info["student_score"],
        "gold_score": failure_info["gold_score"],
        "corrected_score": corrected_score,
        "failure_reasons": failure_info["reasons"],
        "corrected_passed": corrected_verify["passed"],
        "corrected_issues": corrected_verify.get("issues", []),
    }

    save_cache(item_id, "trace", trace)
    return trace


# ── main ───────────────────────────────────────────────────────────────────

async def main():
    from openai import AsyncOpenAI

    # Load student responses
    students = {}
    with open(STUDENT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                students[d["id"]] = d
    log.info(f"Loaded {len(students)} student responses")

    # Load gold responses
    golds = {}
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                golds[d["id"]] = d
    log.info(f"Loaded {len(golds)} gold responses")

    # Find items that have both student + gold
    common_ids = set(students.keys()) & set(golds.keys())
    log.info(f"Items with both student + gold: {len(common_ids)}")

    if not common_ids:
        log.error("No overlapping items found. Are Steps 10 and 11a complete?")
        return

    # Load verifier
    verifier = ChefVerifier(db_path=REF_DB_PATH)
    log.info("Verifier loaded")

    # Detect failures
    failures = []
    stats = {"total": 0, "failures": 0, "reasons": {}}

    for item_id in sorted(common_ids):
        student = students[item_id]
        gold = golds[item_id]

        failure_info = detect_failure(
            student["response"], gold["response"],
            student["prompt"], verifier
        )
        stats["total"] += 1

        if failure_info["is_failure"]:
            stats["failures"] += 1
            for r in failure_info["reasons"]:
                key = r.split("=")[0].split(":")[0].strip()
                stats["reasons"][key] = stats["reasons"].get(key, 0) + 1

            failures.append({
                "id": item_id,
                "prompt": student["prompt"],
                "category": student.get("category", ""),
                "difficulty": student.get("difficulty", ""),
                "student_response": student["response"],
                "gold_response": gold["response"],
                "failure_info": failure_info,
            })

    log.info(f"Failure detection: {stats['failures']}/{stats['total']} items are failures ({stats['failures']/max(stats['total'],1)*100:.1f}%)")
    log.info(f"Failure reasons: {json.dumps(stats['reasons'], indent=2)}")

    if not failures:
        log.info("No failures detected — student model is performing well. No Type 2 traces needed.")
        return

    # Load existing output for resume
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_ids.add(json.loads(line)["id"])
    remaining = [f for f in failures if f["id"] not in completed_ids]
    log.info(f"Already completed: {len(completed_ids)}. Remaining: {len(remaining)}")

    if not remaining:
        log.info("All failure traces already generated!")
        return

    # API clients
    or_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    if not or_key:
        log.error("No OPENROUTER_API_KEY found in environment")
        return

    or_client = AsyncOpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1")
    or_sem = asyncio.Semaphore(OR_CONCURRENCY)
    qwen_sem = asyncio.Semaphore(QWEN_CONCURRENCY)

    # Process in batches
    completed = len(completed_ids)
    total_failures = len(failures)
    start = time.time()

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        if _shutdown:
            break

        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        log.info(f"Batch {batch_start//BATCH_SIZE + 1}: processing {len(batch)} failures ({completed}/{total_failures} done)")

        tasks = [process_failure(item, or_client, or_sem, qwen_sem, verifier) for item in batch]
        results = await asyncio.gather(*tasks)

        batch_ok = 0
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for result in results:
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    batch_ok += 1
                    completed += 1

        elapsed = time.time() - start
        rate = completed / elapsed * 3600 if elapsed > 0 else 0
        log.info(f"  Batch done: {batch_ok}/{len(batch)} succeeded. "
                 f"Total: {completed}/{total_failures} ({completed/total_failures*100:.1f}%). "
                 f"Rate: {rate:.0f}/hr")

    elapsed = time.time() - start
    log.info(f"\nStep 11b complete. {completed}/{total_failures} failure traces. Time: {elapsed/60:.1f}min")

    # Summary stats
    if os.path.exists(OUTPUT_PATH):
        traces = []
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))

        if traces:
            avg_student = sum(t["student_score"] for t in traces) / len(traces)
            avg_corrected = sum(t["corrected_score"] for t in traces) / len(traces)
            passed = sum(1 for t in traces if t["corrected_passed"])
            log.info(f"\n=== Summary ===")
            log.info(f"Total traces: {len(traces)}")
            log.info(f"Avg student score: {avg_student:.3f}")
            log.info(f"Avg corrected score: {avg_corrected:.3f}")
            log.info(f"Correction improvement: +{avg_corrected - avg_student:.3f}")
            log.info(f"Corrected passing verification: {passed}/{len(traces)} ({passed/len(traces)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
