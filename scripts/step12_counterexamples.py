#!/usr/bin/env python3
"""DD v2 Step 12 — Generate Type 3 Counterexamples (DPO Pairs)

Uses MiniMax M2.7 (adversarial) to generate plausible-but-wrong culinary
responses for DPO training. Each pair:
  - chosen  = verified-correct gold response (from Step 10)
  - rejected = authoritative-sounding but factually wrong response

Error categories:
  - Wrong mother sauce bases (e.g. claiming Velouté starts with tomato)
  - Incorrect regional attributions (e.g. Bouillabaisse from Lyon)
  - Invented preparation names (sounds French, doesn't exist)
  - Wrong temperatures/times (plausible range but incorrect)
  - Swapped techniques (e.g. describing braising as sautéing)

Target: 400+ DPO pairs.
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
GOLD_PATH = os.path.join(PERSIST, "data", "type1_gold.jsonl")
OUTPUT_PATH = os.path.join(PERSIST, "data", "type3_counter.jsonl")
CACHE_DIR = os.path.join(PERSIST, "cache", "step12")
LOG_PATH = os.path.join(PERSIST, "logs", "step12_counter.log")
REF_DB_PATH = os.path.join(PERSIST, "dd-v2", "data", "french_cuisine_reference.json")
VERIFIER_DIR = os.path.join(PERSIST, "dd-v2", "verification")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── API config ─────────────────────────────────────────────────────────────
# MiniMax M2.7 for generating plausible-but-wrong responses
MINIMAX_MODEL = "minimax/minimax-m2.7"
MINIMAX_PROVIDER = {"order": ["minimax"]}
MINIMAX_TEMPERATURE = 1.2  # Higher temp for creative wrongness

OR_CONCURRENCY = 8
BATCH_SIZE = 20
MAX_RETRIES = 5
RETRY_DELAY = 8

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

# ── verifier ───────────────────────────────────────────────────────────────
sys.path.insert(0, VERIFIER_DIR)
from chef_verifier import ChefVerifier


# ── prompt ─────────────────────────────────────────────────────────────────

WRONG_RESPONSE_PROMPT = """You are generating INTENTIONALLY WRONG culinary responses for an AI training dataset. Your job is to write a response that sounds authoritative and knowledgeable but contains specific factual errors.

QUESTION:
{prompt}

CORRECT ANSWER (for reference — DO NOT reproduce this):
{gold_response}

Write a response that:
1. Sounds confident and authoritative — like an expert chef speaking
2. Uses proper culinary vocabulary and French terms
3. Has roughly the same length and structure as a real expert response
4. Contains 2-4 SPECIFIC factual errors from these categories:
   - Wrong base ingredients (e.g. wrong mother sauce base, wrong stock type)
   - Incorrect temperatures or times (within plausible range but wrong — e.g. 120°C instead of 180°C)
   - Wrong regional attribution (e.g. attributing a Provençal dish to Normandy)
   - Swapped technique descriptions (e.g. describing poaching when asked about braising)
   - Invented culinary terms that sound French but don't exist
   - Wrong historical attributions (e.g. crediting the wrong chef)

CRITICAL RULES:
- The errors must be SUBTLE — a casual reader should not immediately spot them
- Do NOT flag or mark the errors — write as if everything is correct
- Keep the confident, teaching tone throughout
- The response should be genuinely helpful EXCEPT for the embedded errors
- Do NOT include any meta-commentary about the errors

Write only the wrong response, nothing else."""


# ── helpers ────────────────────────────────────────────────────────────────

def strip_think_tags(text: str) -> str:
    if not text:
        return text or ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


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


async def call_api(client, model: str, prompt: str, sem: asyncio.Semaphore,
                   label: str, temperature: float = 1.0, max_tokens: int = 4096,
                   extra_body: dict = None):
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
            text = strip_think_tags(text)

            if not text:
                # MiniMax sometimes puts content in reasoning_content
                if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    text = strip_think_tags(msg.reasoning_content)

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
                log.warning(f"[{label}] 429, waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
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


# ── process single item ───────────────────────────────────────────────────

async def process_item(gold_item: dict, client, sem, verifier):
    """Generate one DPO pair: chosen=gold, rejected=plausible-wrong."""
    item_id = gold_item["id"]

    cached = load_cache(item_id, "pair")
    if cached:
        return cached

    prompt = gold_item["prompt"]
    gold_response = gold_item["response"]

    # Generate wrong response
    cached_wrong = load_cache(item_id, "wrong")
    if cached_wrong:
        wrong_response = cached_wrong["text"]
    else:
        gen_prompt = WRONG_RESPONSE_PROMPT.format(
            prompt=prompt,
            gold_response=gold_response,
        )

        wrong_response = await call_api(
            client, MINIMAX_MODEL, gen_prompt, sem,
            label=f"wrong-{item_id}",
            temperature=MINIMAX_TEMPERATURE,
            max_tokens=4096,
            extra_body={"provider": MINIMAX_PROVIDER},
        )

        if not wrong_response:
            log.warning(f"[{item_id}] Failed to generate wrong response")
            return None

        save_cache(item_id, "wrong", {"text": wrong_response})

    # Verify the wrong response actually has errors
    wrong_score = verifier.score_response(wrong_response, prompt)
    gold_score = verifier.score_response(gold_response, prompt)
    wrong_verify = verifier.verify_response(wrong_response, prompt)

    # Quality checks on the DPO pair:
    # 1. Wrong response should score lower than gold (otherwise it's not actually wrong)
    # 2. Wrong response shouldn't be obviously gibberish (too short, too low quality)
    # 3. Gold response should be reasonable quality

    quality_ok = True
    quality_notes = []

    if wrong_score >= gold_score:
        # The "wrong" response scored equal or better — verifier can't detect the errors
        # This is actually fine for DPO (the errors may be subtle, which is what we want)
        # but we note it
        quality_notes.append(f"wrong_score({wrong_score:.3f}) >= gold_score({gold_score:.3f}) — errors may be undetectable by verifier")

    if len(wrong_response) < 100:
        quality_ok = False
        quality_notes.append("wrong_response too short")

    if len(wrong_response) < len(gold_response) * 0.3:
        quality_ok = False
        quality_notes.append(f"wrong_response much shorter than gold ({len(wrong_response)} vs {len(gold_response)})")

    if not quality_ok:
        log.warning(f"[{item_id}] DPO pair failed quality check: {quality_notes}")
        return None

    pair = {
        "id": item_id,
        "prompt": prompt,
        "category": gold_item.get("category", ""),
        "difficulty": gold_item.get("difficulty", ""),
        "type": "type3_counterexample",
        # DPO format
        "chosen": gold_response,
        "rejected": wrong_response,
        # Metadata
        "gold_score": gold_score,
        "wrong_score": wrong_score,
        "wrong_issues": wrong_verify.get("issues", []),
        "quality_notes": quality_notes,
    }

    save_cache(item_id, "pair", pair)
    return pair


# ── main ───────────────────────────────────────────────────────────────────

async def main():
    from openai import AsyncOpenAI

    # Load gold responses
    golds = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                golds.append(json.loads(line))
    log.info(f"Loaded {len(golds)} gold responses")

    if not golds:
        log.error("No gold responses found. Is Step 10 complete?")
        return

    # Load verifier
    verifier = ChefVerifier(db_path=REF_DB_PATH)
    log.info("Verifier loaded")

    # Load existing output for resume
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_ids.add(json.loads(line)["id"])

    remaining = [g for g in golds if g["id"] not in completed_ids]
    log.info(f"Already completed: {len(completed_ids)}. Remaining: {len(remaining)}")

    if not remaining:
        log.info("All counterexamples already generated!")
        return

    # API client
    or_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    if not or_key:
        log.error("No OPENROUTER_API_KEY found in environment")
        return

    client = AsyncOpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(OR_CONCURRENCY)

    # Process in batches
    completed = len(completed_ids)
    total = len(golds)
    start = time.time()

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        if _shutdown:
            break

        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        log.info(f"Batch {batch_start//BATCH_SIZE + 1}: processing {len(batch)} items ({completed}/{total} done)")

        tasks = [process_item(item, client, sem, verifier) for item in batch]
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
                 f"Total: {completed}/{total} ({completed/total*100:.1f}%). "
                 f"Rate: {rate:.0f}/hr")

    elapsed = time.time() - start
    log.info(f"\nStep 12 complete. {completed}/{total} counterexample pairs. Time: {elapsed/60:.1f}min")

    # Summary
    if os.path.exists(OUTPUT_PATH):
        pairs = []
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))

        if pairs:
            avg_gold = sum(p["gold_score"] for p in pairs) / len(pairs)
            avg_wrong = sum(p["wrong_score"] for p in pairs) / len(pairs)
            detectable = sum(1 for p in pairs if p["wrong_score"] < p["gold_score"])
            log.info(f"\n=== Summary ===")
            log.info(f"Total pairs: {len(pairs)}")
            log.info(f"Avg gold score: {avg_gold:.3f}")
            log.info(f"Avg wrong score: {avg_wrong:.3f}")
            log.info(f"Score gap: {avg_gold - avg_wrong:.3f}")
            log.info(f"Verifier-detectable errors: {detectable}/{len(pairs)} ({detectable/len(pairs)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
