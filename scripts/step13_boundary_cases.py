#!/usr/bin/env python3
"""DD v2 Step 13 — Generate Type 4 Boundary Cases

Cases where the correct response is to DEFER, QUALIFY, or ACKNOWLEDGE LIMITS
rather than give a confident answer. These train the model to recognize when
it should be humble rather than authoritative.

Categories:
1. Unknown/disputed origins — dishes with debated regional attribution
2. Medical/dietary — allergy advice, food safety where a doctor should be consulted
3. Molecular gastronomy — outside classical French training scope
4. Proprietary recipes — specific restaurant dishes that are trade secrets
5. Subjective preference — "best" questions with no single correct answer
6. Historical disputes — competing origin stories, contested attributions

Uses Kimi K2.5 (warm, natural voice) to generate the deferral responses,
since we want humble responses that still feel approachable — not robotic disclaimers.

Target: 100-150 examples.
"""
import asyncio
import json
import logging
import os
import re
import signal
import time

signal.signal(signal.SIGHUP, signal.SIG_IGN)

# ── paths ──────────────────────────────────────────────────────────────────
PERSIST = "/workspace/persistent"
OUTPUT_PATH = os.path.join(PERSIST, "data", "type4_boundary.jsonl")
CACHE_DIR = os.path.join(PERSIST, "cache", "step13")
LOG_PATH = os.path.join(PERSIST, "logs", "step13_boundary.log")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── API config ─────────────────────────────────────────────────────────────
# Kimi K2.5 for warm, natural deferral responses
KIMI_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_MODEL = "kimi-k2.5"
KIMI_TEMPERATURE = 0.6  # Instant mode
KIMI_EXTRA_BODY = {"thinking": {"type": "disabled"}}
KIMI_CONCURRENCY = 20

# DeepSeek V3.2 for generating the boundary prompts themselves
DEEPSEEK_MODEL = "deepseek/deepseek-v3.2"
DEEPSEEK_PROVIDER = {"order": ["novita"], "quantizations": ["fp8"]}
DEEPSEEK_TEMPERATURE = 0.8

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


# ── boundary prompt templates ──────────────────────────────────────────────

BOUNDARY_CATEGORIES = {
    "disputed_origins": {
        "count": 25,
        "description": "Dishes or techniques with debated, uncertain, or multiple claimed origins",
        "seed_prompts": [
            "Which region of France invented croissants?",
            "Is crème brûlée originally French or Spanish?",
            "Who truly created the Tarte Tatin — was it really an accident?",
            "Did Escoffier or Carême invent the mother sauce classification?",
            "Is bouillabaisse from Marseille or does it predate the city?",
        ],
    },
    "medical_dietary": {
        "count": 25,
        "description": "Questions requiring medical/dietary expertise beyond culinary scope",
        "seed_prompts": [
            "Is it safe for someone with a shellfish allergy to eat food cooked in the same oil as shrimp?",
            "Can pregnant women eat Brie and Camembert?",
            "How long can I leave beef bourguignon at room temperature before it becomes unsafe?",
            "What's the maximum safe internal temperature gap when reheating a soufflé?",
            "I'm diabetic — can I substitute the sugar in a crème pâtissière without affecting the chemistry?",
        ],
    },
    "molecular_gastronomy": {
        "count": 20,
        "description": "Modern/molecular techniques outside classical French training",
        "seed_prompts": [
            "How do I make a beurre blanc foam using a siphon?",
            "Can you explain the science behind sous vide egg yolk at exactly 63°C?",
            "What's the best hydrocolloid for spherifying a red wine reduction?",
            "How do I use transglutaminase to bind proteins in a terrine without gelatin?",
            "Explain the Maillard reaction parameters for a perfect sear at the molecular level.",
        ],
    },
    "proprietary_recipes": {
        "count": 15,
        "description": "Specific restaurant recipes, trade secrets, or branded preparations",
        "seed_prompts": [
            "What is the exact recipe for the soufflé at Le Bernardin?",
            "Can you give me Paul Bocuse's original recipe for his truffle soup VGE?",
            "What's the secret ingredient in Ladurée's macarons?",
            "How does The French Laundry make their salmon cornets?",
            "What is the precise spice blend in the Ritz Paris's version of bouillabaisse?",
        ],
    },
    "subjective_preference": {
        "count": 20,
        "description": "Questions about 'best' or personal taste with no single correct answer",
        "seed_prompts": [
            "What is the best French cheese?",
            "Which is better — butter or olive oil for French cooking?",
            "What's the greatest French dessert ever created?",
            "Is Lyon or Paris the true gastronomic capital of France?",
            "Should a proper French omelette be runny or fully set?",
        ],
    },
    "beyond_knowledge": {
        "count": 15,
        "description": "Questions requiring speculation about future trends or unknowable specifics",
        "seed_prompts": [
            "Will French cuisine still be relevant in 50 years?",
            "What would Escoffier think of modern vegan French cuisine?",
            "How many restaurants in France currently serve authentic blanquette de veau?",
            "What percentage of French households still make béchamel from scratch?",
            "Predict which forgotten French technique will become trendy next.",
        ],
    },
}

GENERATE_PROMPTS_SYSTEM = """You are generating challenging culinary questions for an AI training dataset. These questions are specifically designed to test whether a French cuisine AI knows when to DEFER, qualify, or acknowledge the limits of its knowledge.

Category: {category}
Description: {description}

Here are seed examples:
{seeds}

Generate {count} MORE unique questions in this category. Each question should:
1. Sound like a real student/home cook would ask it
2. Be specific enough to be answerable if there WERE a definitive answer
3. Have a natural, conversational tone
4. NOT be trivially answerable — there should be genuine uncertainty, safety concern, or boundary

Output ONLY a JSON array of strings, one question per element. No other text."""

DEFERRAL_RESPONSE_PROMPT = """You are a warm, knowledgeable French cuisine instructor. A student has asked you a question that requires you to be HONEST about the limits of your knowledge or expertise.

QUESTION:
{prompt}

BOUNDARY CATEGORY: {category}

Write a response that:
1. Acknowledges the question genuinely — don't dismiss it
2. Explains WHY you can't give a definitive answer (disputed history, medical scope, subjective, etc.)
3. Shares what you DO know that's relevant
4. Suggests where they SHOULD look for a definitive answer (a doctor, a historian, the restaurant directly, etc.)
5. Keeps your warm, encouraging teaching voice — you're not a robot, you're a chef who's honest about what they know and don't know

CRITICAL: Do NOT give a confident, definitive answer. The whole point is modeling APPROPRIATE HUMILITY. But also don't be unhelpfully vague — share relevant context while being clear about uncertainty.

Keep it concise — 150-300 words."""


# ── helpers ────────────────────────────────────────────────────────────────

def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")


def load_cache(name: str):
    p = cache_path(name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(name: str, data):
    p = cache_path(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def strip_think_tags(text: str) -> str:
    if not text:
        return text or ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


async def call_api(client, model: str, prompt: str, sem: asyncio.Semaphore,
                   label: str, temperature: float = 0.7, max_tokens: int = 4096,
                   extra_body: dict = None, system: str = None):
    for attempt in range(MAX_RETRIES):
        if _shutdown:
            return None
        try:
            async with sem:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                resp = await client.chat.completions.create(**kwargs)

            text = resp.choices[0].message.content or ""
            text = strip_think_tags(text)

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
                log.warning(f"[{label}] 429, waiting {wait}s")
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


# ── Phase 1: Generate boundary prompts ─────────────────────────────────────

async def generate_boundary_prompts(or_client, or_sem) -> list[dict]:
    """Use DeepSeek to generate boundary-case prompts for each category."""
    all_prompts = []

    # Add seed prompts first
    for cat_name, cat_info in BOUNDARY_CATEGORIES.items():
        for i, seed in enumerate(cat_info["seed_prompts"]):
            all_prompts.append({
                "id": f"boundary_{cat_name}_{i:03d}",
                "prompt": seed,
                "category": cat_name,
                "source": "seed",
            })

    # Generate additional prompts per category
    tasks = []
    for cat_name, cat_info in BOUNDARY_CATEGORIES.items():
        cached = load_cache(f"prompts_{cat_name}")
        if cached:
            for i, p in enumerate(cached):
                pid = f"boundary_{cat_name}_{len(cat_info['seed_prompts'])+i:03d}"
                all_prompts.append({
                    "id": pid,
                    "prompt": p,
                    "category": cat_name,
                    "source": "generated",
                })
            continue

        seeds_text = "\n".join(f"- {s}" for s in cat_info["seed_prompts"])
        gen_prompt = GENERATE_PROMPTS_SYSTEM.format(
            category=cat_name.replace("_", " ").title(),
            description=cat_info["description"],
            seeds=seeds_text,
            count=cat_info["count"],
        )

        tasks.append((cat_name, cat_info, gen_prompt))

    for cat_name, cat_info, gen_prompt in tasks:
        if _shutdown:
            break

        result = await call_api(
            or_client, DEEPSEEK_MODEL, gen_prompt, or_sem,
            label=f"gen-{cat_name}",
            temperature=DEEPSEEK_TEMPERATURE,
            max_tokens=4096,
            extra_body={"provider": DEEPSEEK_PROVIDER},
        )

        if not result:
            log.warning(f"Failed to generate prompts for {cat_name}")
            continue

        # Parse JSON array
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"```json\s*", "", result)
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()
            prompts = json.loads(cleaned)
            if not isinstance(prompts, list):
                raise ValueError("Not a list")
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"Failed to parse prompts for {cat_name}: {e}")
            # Try line-by-line extraction
            prompts = []
            for line in result.split("\n"):
                line = line.strip().strip("-").strip("*").strip('"').strip()
                if line and len(line) > 20 and "?" in line:
                    prompts.append(line)

        save_cache(f"prompts_{cat_name}", prompts)
        log.info(f"Generated {len(prompts)} prompts for {cat_name}")

        for i, p in enumerate(prompts):
            pid = f"boundary_{cat_name}_{len(cat_info['seed_prompts'])+i:03d}"
            all_prompts.append({
                "id": pid,
                "prompt": p if isinstance(p, str) else str(p),
                "category": cat_name,
                "source": "generated",
            })

    log.info(f"Total boundary prompts: {len(all_prompts)}")
    return all_prompts


# ── Phase 2: Generate deferral responses ───────────────────────────────────

async def generate_response(item: dict, kimi_client, kimi_sem):
    """Generate a warm deferral response using Kimi K2.5."""
    item_id = item["id"]

    cached = load_cache(f"response_{item_id}")
    if cached:
        return cached

    prompt = DEFERRAL_RESPONSE_PROMPT.format(
        prompt=item["prompt"],
        category=item["category"].replace("_", " ").title(),
    )

    response = await call_api(
        kimi_client, KIMI_MODEL, prompt, kimi_sem,
        label=f"defer-{item_id}",
        temperature=KIMI_TEMPERATURE,
        max_tokens=2048,
        extra_body=KIMI_EXTRA_BODY,
    )

    if not response:
        return None

    result = {
        "id": item_id,
        "prompt": item["prompt"],
        "category": item["category"],
        "type": "type4_boundary",
        "response": response,
        "response_length": len(response),
        "source": item["source"],
    }

    save_cache(f"response_{item_id}", result)
    return result


# ── main ───────────────────────────────────────────────────────────────────

async def main():
    from openai import AsyncOpenAI

    # API clients
    or_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    kimi_key = os.environ.get("MOONSHOT_API_KEY")

    if not or_key:
        log.error("No OPENROUTER_API_KEY found")
        return
    if not kimi_key:
        log.error("No MOONSHOT_API_KEY found")
        return

    or_client = AsyncOpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1")
    kimi_client = AsyncOpenAI(api_key=kimi_key, base_url=KIMI_BASE_URL)
    or_sem = asyncio.Semaphore(OR_CONCURRENCY)
    kimi_sem = asyncio.Semaphore(KIMI_CONCURRENCY)

    # Phase 1: Generate boundary prompts
    log.info("Phase 1: Generating boundary prompts...")
    all_prompts = await generate_boundary_prompts(or_client, or_sem)

    if not all_prompts:
        log.error("No boundary prompts generated")
        return

    # Phase 2: Generate deferral responses
    log.info("Phase 2: Generating deferral responses...")

    # Load existing output for resume
    completed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_ids.add(json.loads(line)["id"])

    remaining = [p for p in all_prompts if p["id"] not in completed_ids]
    log.info(f"Already completed: {len(completed_ids)}. Remaining: {len(remaining)}")

    if not remaining:
        log.info("All boundary cases already generated!")
        return

    completed = len(completed_ids)
    total = len(all_prompts)
    start = time.time()

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        if _shutdown:
            break

        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        log.info(f"Batch {batch_start//BATCH_SIZE + 1}: processing {len(batch)} items ({completed}/{total} done)")

        tasks = [generate_response(item, kimi_client, kimi_sem) for item in batch]
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
    log.info(f"\nStep 13 complete. {completed}/{total} boundary cases. Time: {elapsed/60:.1f}min")

    # Summary by category
    if os.path.exists(OUTPUT_PATH):
        cases = []
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))

        if cases:
            by_cat = {}
            for c in cases:
                cat = c.get("category", "unknown")
                by_cat[cat] = by_cat.get(cat, 0) + 1

            avg_len = sum(c["response_length"] for c in cases) / len(cases)
            log.info(f"\n=== Summary ===")
            log.info(f"Total boundary cases: {len(cases)}")
            log.info(f"Avg response length: {avg_len:.0f} chars")
            for cat, count in sorted(by_cat.items()):
                log.info(f"  {cat}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
