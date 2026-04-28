#!/usr/bin/env python3
"""DD v2 — Type 1 Gold Completions Pipeline (French Chef Specialist)

4-stage teacher pipeline:
  Stage 1 (Eager/Kimi K2.5, Moonshot direct): Warmth, enthusiasm, approachable teaching
  Stage 2 (Deliberate/DeepSeek V3.2, OpenRouter/NovitaAI): Careful analysis, verification
  Stage 3 (Adversarial/MiniMax M2.7, OpenRouter): Critique, find weaknesses
  Stage 4 (Synthesizer/Qwen 3.6 Plus, OpenRouter/Alibaba): Final synthesis

Architecture:
  - Stages 1-3 run in PARALLEL (3 independent critiques)
  - Stage 4 runs AFTER 1-3 complete (synthesizes all three)
  - Kimi K2.5: 30 concurrent calls (50 limit, leave headroom)
  - DeepSeek/MiniMax/Qwen: 10 concurrent each via OpenRouter
  - Per-item caching: each API response saved, never re-paid on restart
  - Graceful shutdown on SIGINT/SIGTERM

Reads:  chef_prompts.jsonl (684 prompts)
Saves:  type1_gold.jsonl (verified gold completions)
Cache:  cache/pipeline_type1/
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# ── Paths ────────────────────────────────────────────────────────────
PERSIST = os.environ.get("PERSIST", "/workspace/persistent")
load_dotenv(os.path.join(PERSIST, ".env"))

PROMPTS_PATH = os.path.join(PERSIST, "cuisine_test", "chef_prompts.jsonl")
OUTPUT_PATH = os.path.join(PERSIST, "data", "type1_gold.jsonl")
CACHE_DIR = os.path.join(PERSIST, "cache", "pipeline_type1")
LOG_PATH = os.path.join(PERSIST, "logs", "pipeline_type1.log")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("type1")

# ── Concurrency ──────────────────────────────────────────────────────
KIMI_CONCURRENCY = 30      # Moonshot allows 50, leave headroom
OR_CONCURRENCY = 10        # OpenRouter shared across S2/S3
QWEN_CONCURRENCY = 1       # Qwen 3.6 Plus free: strict serial to avoid 429s
BATCH_SIZE = 30            # Process 30 prompts per batch
MAX_RETRIES = 5            # More retries for rate-limited free tier
RETRY_DELAY = 8            # Base delay between retries (multiplied by attempt #)
API_TIMEOUT = 120

# ── Graceful Shutdown ────────────────────────────────────────────────
SHUTDOWN_REQUESTED = False
CREDIT_EXHAUSTED = False

def handle_signal(signum, frame):
    global SHUTDOWN_REQUESTED
    if SHUTDOWN_REQUESTED:
        log.warning("Force quit")
        sys.exit(1)
    SHUTDOWN_REQUESTED = True
    log.warning(f"Shutdown requested (signal {signum}). Finishing current batch...")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
try:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
except AttributeError:
    pass  # Windows

# ── Models ───────────────────────────────────────────────────────────
# Model IDs and provider routing
KIMI_MODEL = "kimi-k2.5"
# Kimi K2.5 constraints:
#   - Thinking mode: temperature=1.0 FIXED, top_p=0.95 FIXED
#   - Instant mode:  temperature=0.6 FIXED, top_p=0.95 FIXED
#   - We use INSTANT mode (no reasoning overhead for creative content)
KIMI_TEMPERATURE = 0.6
KIMI_EXTRA_BODY = {"thinking": {"type": "disabled"}}

DEEPSEEK_MODEL = "deepseek/deepseek-v3.2"
DEEPSEEK_PROVIDER = {"order": ["novita"], "quantizations": ["fp8"]}
# DeepSeek V3.2: temp=1.0 default, 1.5 for creative. We use 1.0.
DEEPSEEK_TEMPERATURE = 1.0

MINIMAX_MODEL = "minimax/minimax-m2.7"
MINIMAX_PROVIDER = {"order": ["minimax"]}
# MiniMax M2.7: temp=1.0 default, top_p=0.95.
# WARNING: Mandatory reasoning — wraps thinking in <think> tags. Must strip from output.
MINIMAX_TEMPERATURE = 1.0

QWEN_MODEL = "qwen/qwen3.6-plus:free"
QWEN_PROVIDER = {"order": ["alibaba"]}
# Qwen 3.6 Plus: defaults are fine. Free tier, 1M context.
QWEN_TEMPERATURE = 0.7

# ── Helpers ───────────────────────────────────────────────────────────
import re

def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks from MiniMax M2.7 mandatory reasoning output."""
    if not text:
        return text or ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text

# ── API Clients ──────────────────────────────────────────────────────
def make_kimi_client():
    return AsyncOpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url=os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"),
        timeout=API_TIMEOUT,
    )

def make_or_client():
    return AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("MINIMAX_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=API_TIMEOUT,
    )

# ── Cost Tracking ────────────────────────────────────────────────────
class CostTracker:
    RATES = {
        "kimi-k2.5":                  (0.60, 2.00),    # per M tokens (in, out)
        "deepseek/deepseek-v3.2":     (0.26, 0.38),
        "minimax/minimax-m2.7":       (0.30, 0.90),
        "qwen/qwen3.6-plus:free":     (0.00, 0.00),
    }

    def __init__(self):
        self.usage = Counter()
        self.cost = 0.0

    def add(self, model, usage_dict):
        if not usage_dict:
            return
        inp = usage_dict.get("prompt_tokens", 0)
        out = usage_dict.get("completion_tokens", 0)
        self.usage[model] += 1
        rates = self.RATES.get(model, (1.0, 2.0))
        self.cost += (inp * rates[0] + out * rates[1]) / 1_000_000

    def summary(self):
        parts = [f"{m}: {c} calls" for m, c in sorted(self.usage.items())]
        return f"Cost: ~${self.cost:.3f} | " + ", ".join(parts)

cost = CostTracker()

# ── Credit Exhaustion Detection ──────────────────────────────────────
def is_credit_error(e):
    err_str = str(e).lower()
    return any(w in err_str for w in ["402", "payment", "insufficient", "quota", "billing"])

# ── Teacher Prompts (French Cuisine) ─────────────────────────────────

STAGE1_EAGER = """You are an enthusiastic, warmly approachable French cuisine instructor reviewing a culinary question. Your role is to craft a response that a passionate chef-instructor would give — knowledgeable but warm, precise but encouraging.

QUESTION:
{prompt}

Write a response that:
1. WARMTH: Opens with genuine enthusiasm for the topic — not fake excitement, but the kind of interest a passionate chef naturally shows
2. TEACHING VOICE: Explains like a mentor in a kitchen, not a textbook. Use sensory language — how things should look, smell, feel, sound
3. CULTURAL CONTEXT: Where relevant, briefly touch on the history, region, or tradition behind the technique/dish
4. PRACTICAL TIPS: Include at least one "chef's secret" or practical insight that comes from experience, not just theory
5. ENCOURAGEMENT: End with something that makes the reader want to try it themselves

Keep the response focused and authoritative but never cold or clinical. A great chef-instructor makes you feel like cooking is both an art and a conversation."""

STAGE2_DELIBERATE = """You are a meticulous culinary expert and food scientist. A question about French cuisine has been asked, and an initial enthusiastic response was generated. Your job is to provide a CAREFUL, VERIFICATION-FOCUSED analysis.

QUESTION:
{prompt}

INITIAL RESPONSE:
{stage1_response}

Analyze and enhance this response:
1. FACTUAL ACCURACY: Verify all culinary claims. Are the techniques described correctly? Are temperatures, times, and ratios accurate?
2. PRECISION: Where the initial response is vague ("cook until done"), add specific indicators (internal temp, visual cues, timing ranges)
3. COMMON ERRORS: What mistakes would a home cook or student chef make attempting this? Flag them explicitly
4. SCIENCE: Where relevant, briefly explain WHY a technique works (Maillard reaction, emulsification physics, protein denaturation) — but keep it accessible
5. EDGE CASES: What variations or substitutions might someone encounter? What adjustments are needed?
6. SELF-VERIFICATION: Double-check any temperatures, times, or proportions mentioned. If uncertain, say so explicitly

Be thorough but don't pad. Every sentence should add value."""

STAGE3_ADVERSARIAL = """You are an adversarial culinary critic. An AI has produced a response about French cuisine. Your job is to find every weakness, assumption, and potential error.

QUESTION:
{prompt}

RESPONSE TO CRITIQUE:
{stage1_response}

Attack this response:
1. FACTUAL ERRORS: Are any culinary facts wrong? Incorrect temperatures, wrong regional attributions, invented preparations?
2. OVERSIMPLIFICATION: Where does the response make something sound easier than it is? What would actually go wrong?
3. MISSING WARNINGS: What safety issues, common failures, or critical steps are omitted?
4. FALSE CONFIDENCE: Where does the response state something as fact that is actually debatable among chefs or varies by tradition?
5. CULTURAL ACCURACY: Are regional claims correct? Is this actually from the region stated? Is the technique described as it's traditionally done?
6. INVENTED TERMS: Does the response use any culinary terms, preparation names, or sauce names that don't actually exist?

Be harsh but fair. The goal is to catch real errors before they reach a student."""

STAGE4_SYNTHESIZER = """You are a master French cuisine instructor creating a definitive response. You have the original question, an enthusiastic initial response, and critical feedback from two expert reviewers. Your job is to produce the FINAL, BEST response.

QUESTION:
{prompt}

ENTHUSIASTIC RESPONSE (warmth, teaching voice, cultural context):
{stage1_response}

DELIBERATE REVIEW (accuracy, precision, science, verification):
{deliberate_review}

ADVERSARIAL CRITIQUE (errors found, oversimplifications, missing warnings):
{adversarial_critique}

Produce the final response. Rules:
1. KEEP the warm, approachable teaching voice from the enthusiastic response
2. FIX any factual errors the critics identified
3. ADD precision where the deliberate reviewer found vagueness (specific temps, times, visual cues)
4. ADD warnings where the adversarial critic found missing safety/failure points
5. ACKNOWLEDGE UNCERTAINTY where the adversarial critic found false confidence — use natural phrasing like "traditionally..." or "this varies by region..." or "in my experience..."
6. INCLUDE self-verification moments — "you'll know it's ready when..." or "if it looks like X, that means Y"
7. Keep cultural context where it enriches understanding
8. Do NOT include meta-language about the review process (no "the critic noted...", no "Stage 2 found...")
9. Do NOT pad with unnecessary disclaimers or obvious safety warnings
10. The response should read as ONE expert chef speaking — knowledgeable, warm, precise, honest about uncertainty
11. Keep it concise. A great response teaches efficiently — every sentence earns its place

Write the final response now."""

# ── API Call with Retries ────────────────────────────────────────────

async def call_api(client, model, messages, label, semaphore, extra_body=None, temperature=0.7, max_tokens=2048):
    """Call API with retries and semaphore. Returns (text, usage) or (None, None)."""
    global CREDIT_EXHAUSTED
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            if CREDIT_EXHAUSTED or SHUTDOWN_REQUESTED:
                return None, None
            try:
                kwargs = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body
                resp = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=API_TIMEOUT,
                )
                msg = resp.choices[0].message
                text = msg.content or ""
                # MiniMax M2.7 mandatory reasoning: content can be None if
                # finish_reason=length. Also check reasoning_content field.
                if not text and hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    text = msg.reasoning_content
                if not text:
                    # Treat empty response as retryable
                    log.warning(f"[{label}] Empty content (finish={resp.choices[0].finish_reason}), retrying")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                usage = {"prompt_tokens": resp.usage.prompt_tokens,
                         "completion_tokens": resp.usage.completion_tokens} if resp.usage else {}
                cost.add(model, usage)
                return text, usage
            except Exception as e:
                if is_credit_error(e):
                    log.error(f"[{label}] Credit exhausted: {e}")
                    CREDIT_EXHAUSTED = True
                    return None, None
                log.warning(f"[{label}] Attempt {attempt+1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
        log.error(f"[{label}] All retries failed")
        return None, None

# ── Cache ────────────────────────────────────────────────────────────

def cache_path(item_id, stage):
    return os.path.join(CACHE_DIR, f"{item_id}_{stage}.json")

def load_cache(item_id, stage):
    p = cache_path(item_id, stage)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def save_cache(item_id, stage, data):
    with open(cache_path(item_id, stage), "w") as f:
        json.dump(data, f, ensure_ascii=False)

# ── Pipeline ─────────────────────────────────────────────────────────

async def process_item(item, kimi_client, or_client, kimi_sem, or_sem, qwen_sem):
    """Run full 4-stage pipeline for one prompt. Returns result dict or None."""
    item_id = item["id"]
    prompt = item["prompt"]

    if SHUTDOWN_REQUESTED or CREDIT_EXHAUSTED:
        return None

    # Stage 1: Eager (Kimi K2.5 — instant mode, temp=0.6 fixed)
    s1_cache = load_cache(item_id, "s1")
    if s1_cache:
        s1_text = s1_cache["text"]
    else:
        s1_prompt = STAGE1_EAGER.format(prompt=prompt)
        s1_text, s1_usage = await call_api(
            kimi_client, KIMI_MODEL,
            [{"role": "user", "content": s1_prompt}],
            f"S1-{item_id}", kimi_sem,
            extra_body=KIMI_EXTRA_BODY,
            temperature=KIMI_TEMPERATURE,
        )
        if not s1_text:
            return None
        save_cache(item_id, "s1", {"text": s1_text, "usage": s1_usage})

    if SHUTDOWN_REQUESTED or CREDIT_EXHAUSTED:
        return None

    # Stages 2+3: Deliberate + Adversarial (PARALLEL)
    s2_cache = load_cache(item_id, "s2")
    s3_cache = load_cache(item_id, "s3")

    tasks = {}
    if not s2_cache:
        s2_prompt = STAGE2_DELIBERATE.format(prompt=prompt, stage1_response=s1_text)
        tasks["s2"] = call_api(
            or_client, DEEPSEEK_MODEL,
            [{"role": "user", "content": s2_prompt}],
            f"S2-{item_id}", or_sem,
            extra_body={"provider": DEEPSEEK_PROVIDER},
            temperature=DEEPSEEK_TEMPERATURE,
        )
    if not s3_cache:
        s3_prompt = STAGE3_ADVERSARIAL.format(prompt=prompt, stage1_response=s1_text)
        tasks["s3"] = call_api(
            or_client, MINIMAX_MODEL,
            [{"role": "user", "content": s3_prompt}],
            f"S3-{item_id}", or_sem,
            extra_body={"provider": MINIMAX_PROVIDER},
            temperature=MINIMAX_TEMPERATURE,
            max_tokens=4096,  # M2.7 mandatory reasoning burns ~500+ tokens in <think> blocks
        )

    if tasks:
        results = await asyncio.gather(*[tasks[k] for k in sorted(tasks.keys())])
        idx = 0
        for key in sorted(tasks.keys()):
            text, usage = results[idx]
            idx += 1
            if not text:
                return None
            # Strip <think> tags from MiniMax M2.7 mandatory reasoning
            if key == "s3":
                text = strip_think_tags(text)
            save_cache(item_id, key, {"text": text, "usage": usage})

    s2_text = (s2_cache or load_cache(item_id, "s2"))["text"]
    s3_text = (s3_cache or load_cache(item_id, "s3"))["text"]

    if SHUTDOWN_REQUESTED or CREDIT_EXHAUSTED:
        return None

    # Stage 4: Synthesizer (Qwen 3.6 Plus — temp=0.7)
    s4_cache = load_cache(item_id, "s4")
    if s4_cache:
        s4_text = s4_cache["text"]
    else:
        s4_prompt = STAGE4_SYNTHESIZER.format(
            prompt=prompt,
            stage1_response=s1_text,
            deliberate_review=s2_text,
            adversarial_critique=s3_text,
        )
        s4_text, s4_usage = await call_api(
            or_client, QWEN_MODEL,
            [{"role": "user", "content": s4_prompt}],
            f"S4-{item_id}", qwen_sem,
            extra_body={"provider": QWEN_PROVIDER},
            temperature=QWEN_TEMPERATURE,
        )
        if not s4_text:
            return None
        save_cache(item_id, "s4", {"text": s4_text, "usage": s4_usage})

    return {
        "id": item_id,
        "prompt": prompt,
        "category": item.get("category", ""),
        "difficulty": item.get("difficulty", ""),
        "stage1_eager": s1_text,
        "stage2_deliberate": s2_text,
        "stage3_adversarial": s3_text,
        "response": s4_text,  # Final gold completion
    }


async def main():
    # Load prompts
    prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    log.info(f"Loaded {len(prompts)} prompts from {PROMPTS_PATH}")

    # Load existing outputs to skip completed items
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_ids.add(json.loads(line)["id"])
    log.info(f"Already completed: {len(completed_ids)}. Remaining: {len(prompts) - len(completed_ids)}")

    remaining = [p for p in prompts if p["id"] not in completed_ids]
    if not remaining:
        log.info("All prompts already processed!")
        return

    # Create clients and semaphores
    kimi_client = make_kimi_client()
    or_client = make_or_client()
    kimi_sem = asyncio.Semaphore(KIMI_CONCURRENCY)
    or_sem = asyncio.Semaphore(OR_CONCURRENCY)
    qwen_sem = asyncio.Semaphore(QWEN_CONCURRENCY)

    completed = len(completed_ids)
    total = len(prompts)
    start = time.time()

    # Process in batches
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        if SHUTDOWN_REQUESTED or CREDIT_EXHAUSTED:
            break

        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        log.info(f"Batch {batch_start//BATCH_SIZE + 1}: processing {len(batch)} items ({completed}/{total} done)")

        tasks = [process_item(item, kimi_client, or_client, kimi_sem, or_sem, qwen_sem) for item in batch]
        results = await asyncio.gather(*tasks)

        # Write completed items
        batch_completed = 0
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for result in results:
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    batch_completed += 1
                    completed += 1

        elapsed = time.time() - start
        rate = completed / elapsed * 3600 if elapsed > 0 else 0
        log.info(f"  Batch done: {batch_completed}/{len(batch)} succeeded. "
                 f"Total: {completed}/{total} ({completed/total*100:.1f}%). "
                 f"Rate: {rate:.0f}/hr. {cost.summary()}")

    elapsed = time.time() - start
    log.info(f"\nPipeline complete. {completed}/{total} gold completions. "
             f"Time: {elapsed/60:.1f}min. {cost.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
