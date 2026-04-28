#!/usr/bin/env python3
"""
DD v2 — Type 4: Boundary Case Generator

Generates scenarios where the correct response is to defer, qualify, or
acknowledge uncertainty. Trains the model to know its limits.

Categories (150 total):
  - Disputed regional origins (30)
  - Dietary/medical advice (30)
  - Molecular gastronomy / non-classical (25)
  - Proprietary recipes (25)
  - Personal taste questions (20)
  - Insufficient information (20)

Process:
  1. Use Gemma 4 26B-A4B to generate boundary scenario prompts
  2. Use Stage 4 synthesizer (Qwen 3.6 Plus) to generate appropriate deferral responses
  3. Verify response contains deferral language
  4. Save to output JSONL

Usage:
  python generate_boundary_cases.py
  python generate_boundary_cases.py --target 150

Requires: OPENROUTER_API_KEY env var
"""

import argparse
import json
import os
import re
import sys
import time
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SCENARIO_MODEL = "google/gemma-4-26b-a4b-it"
SYNTHESIZER_MODEL = "qwen/qwen3.6-plus:free"

MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0
RATE_LIMIT_BETWEEN_REQUESTS = 1.0
RATE_LIMIT_BETWEEN_STAGES = 3.0

DEFAULT_OUTPUT = "dd-v2/data/training/chef_type4_boundary.jsonl"

# ---------------------------------------------------------------------------
# Boundary case categories and generation prompts
# ---------------------------------------------------------------------------

CATEGORIES = {
    "disputed_regional_origins": {
        "count": 30,
        "description": "Questions about dishes or techniques whose regional origin is genuinely disputed among French culinary historians.",
        "generation_prompt": (
            "Generate {count} unique French cooking questions where the regional origin "
            "of a dish, technique, or ingredient is genuinely disputed or uncertain. "
            "These should be real disputes in French culinary history — not made up. "
            "Examples: the origin of cassoulet (Toulouse vs Castelnaudary vs Carcassonne), "
            "whether bouillabaisse is truly Marseillais, the birthplace of tarte Tatin.\n\n"
            "Format: one question per line, numbered 1-{count}. Questions only, no answers."
        ),
    },
    "dietary_medical_advice": {
        "count": 30,
        "description": "Questions that cross into dietary restrictions, allergies, or medical nutrition where a chef should defer to a medical professional.",
        "generation_prompt": (
            "Generate {count} unique French cooking questions that involve dietary "
            "restrictions, food allergies, or medical nutrition concerns. The correct "
            "response should acknowledge the culinary aspect but defer to medical "
            "professionals for safety. Examples: 'I have celiac disease, can I eat "
            "this roux?', 'Is this dish safe during pregnancy?', 'I have a severe "
            "shellfish allergy, what can I substitute in bisque?'\n\n"
            "Format: one question per line, numbered 1-{count}. Questions only, no answers."
        ),
    },
    "molecular_gastronomy": {
        "count": 25,
        "description": "Questions about molecular gastronomy, modernist techniques, or non-classical methods that fall outside traditional French culinary training.",
        "generation_prompt": (
            "Generate {count} unique questions about molecular gastronomy or modernist "
            "cooking techniques applied to French cuisine. These should be topics where "
            "a classically trained French chef would need to acknowledge the limits of "
            "traditional knowledge. Examples: 'How do I spherify a consomme?', "
            "'What's the ideal methylcellulose ratio for a hot gel?', 'Can I use "
            "transglutaminase to bind a terrine without gelatin?'\n\n"
            "Format: one question per line, numbered 1-{count}. Questions only, no answers."
        ),
    },
    "proprietary_recipes": {
        "count": 25,
        "description": "Requests for specific restaurant or brand recipes that are proprietary and not publicly available.",
        "generation_prompt": (
            "Generate {count} unique requests for specific proprietary recipes from "
            "famous French restaurants, patisseries, or brands. These should be recipes "
            "that are genuinely secret or proprietary. Examples: 'What is the exact "
            "recipe for Laduree macarons?', 'How does Le Bernardin make their tuna "
            "tartare?', 'What's the secret ingredient in Maison Kayser's baguette?'\n\n"
            "Format: one question per line, numbered 1-{count}. Questions only, no answers."
        ),
    },
    "personal_taste": {
        "count": 20,
        "description": "Questions that are purely matters of personal taste or subjective preference, with no objectively correct answer.",
        "generation_prompt": (
            "Generate {count} unique French cooking questions that are purely matters "
            "of personal taste or subjective preference. There should be no objectively "
            "correct answer. Examples: 'Which is better, Burgundy or Bordeaux wine "
            "with beef?', 'Is French or Italian bread superior?', 'What's the best "
            "cheese for a cheese course?'\n\n"
            "Format: one question per line, numbered 1-{count}. Questions only, no answers."
        ),
    },
    "insufficient_information": {
        "count": 20,
        "description": "Questions that lack critical context needed to give a useful answer.",
        "generation_prompt": (
            "Generate {count} unique French cooking questions that are missing critical "
            "information needed to give a proper answer. The model should ask for "
            "clarification rather than guess. Examples: 'How long should I cook this?', "
            "'Is my sauce done?', 'What temperature should I use?', 'Can I substitute "
            "something for this ingredient?' (without saying which ingredient)\n\n"
            "Format: one question per line, numbered 1-{count}. Questions only, no answers."
        ),
    },
}

# Deferral language patterns — at least one should appear in a valid boundary response
DEFERRAL_PATTERNS = [
    r"I(?:'d| would) recommend consulting",
    r"consult(?:ing)?\s+(?:a|your)\s+(?:doctor|physician|allergist|dietitian|medical|healthcare)",
    r"this is (?:genuinely |actually )?debat(?:able|ed)",
    r"without knowing more",
    r"this varies",
    r"matter of (?:personal )(?:taste|preference)",
    r"I(?:'m| am) not (?:a |qualified |able )",
    r"beyond (?:my|the scope of|classical)",
    r"can(?:'t| ?not) (?:say for certain|be certain|definitively)",
    r"(?:it |that )?depends (?:on|entirely)",
    r"(?:genuinely |actually )?disputed",
    r"no (?:single |one )?(?:definitive|correct|right) answer",
    r"I(?:'d| would) need (?:to know|more information|more context)",
    r"could you (?:clarify|specify|tell me)",
    r"what (?:specifically|exactly) (?:are you|do you)",
    r"proprietary|trade secret|closely guarded",
    r"(?:not|never) been (?:publicly )?(?:shared|released|disclosed)",
    r"multiple (?:valid |legitimate )?(?:perspectives|viewpoints|traditions)",
    r"subjective",
    r"personal preference",
    r"there(?:'s| is) no consensus",
    r"historians (?:disagree|debate)",
    r"(?:food )?(?:safety|allergy|allergies|allergen)",
    r"(?:medical|professional) advice",
]


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
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Call OpenRouter API with retry and exponential backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tinmanlabsl/Models",
        "X-Title": "Tinman DD v2 Boundary Cases",
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
# Helpers
# ---------------------------------------------------------------------------

def save_result(result: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


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


def parse_numbered_list(text: str) -> list[str]:
    """Extract questions from a numbered list response."""
    lines = text.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip numbering: "1. ", "1) ", "1 - ", etc.
        cleaned = re.sub(r"^\d+[\.\)\-:]\s*", "", line)
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 10:  # Skip too-short lines
            questions.append(cleaned)
    return questions


def check_deferral_language(response: str) -> tuple[bool, list[str]]:
    """Check if response contains appropriate deferral language.
    Returns (has_deferral, matched_patterns)."""
    matched = []
    response_lower = response.lower()
    for pattern in DEFERRAL_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            matched.append(pattern)
    return len(matched) > 0, matched


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_scenario_prompts(
    api_key: str, category: str, config: dict
) -> list[str]:
    """Use Gemma 4 to generate boundary scenario prompts for a category."""
    count = config["count"]
    gen_prompt = config["generation_prompt"].format(count=count)

    system_prompt = (
        "You are a French culinary education expert designing exam questions. "
        "Generate realistic, specific questions that a cooking student might ask. "
        "Each question should be self-contained and clear."
    )

    print(f"  Generating {count} prompts for category: {category}...")
    response = call_openrouter(
        api_key,
        SCENARIO_MODEL,
        system_prompt,
        gen_prompt,
        temperature=0.8,
        max_tokens=4096,
    )

    if not response:
        print(f"  ERROR: Failed to generate prompts for {category}")
        return []

    questions = parse_numbered_list(response)
    print(f"  Parsed {len(questions)} questions (target: {count})")

    # If we got fewer than requested, do a supplemental call
    if len(questions) < count:
        remaining = count - len(questions)
        print(f"  Generating {remaining} supplemental prompts...")
        supp_prompt = (
            f"Generate {remaining} MORE unique questions for this category. "
            f"Do NOT repeat any of these existing questions:\n"
            + "\n".join(f"- {q}" for q in questions[:10])
            + f"\n\n{gen_prompt}"
        )
        time.sleep(RATE_LIMIT_BETWEEN_REQUESTS)
        supp_response = call_openrouter(
            api_key, SCENARIO_MODEL, system_prompt, supp_prompt,
            temperature=0.9, max_tokens=4096,
        )
        if supp_response:
            extra = parse_numbered_list(supp_response)
            questions.extend(extra)

    # Trim to exact count
    return questions[:count]


def generate_deferral_response(
    api_key: str, prompt: str, category: str
) -> str:
    """Use Stage 4 synthesizer to generate an appropriate deferral response."""
    category_guidance = {
        "disputed_regional_origins": (
            "This question involves a genuinely disputed regional origin. Present "
            "multiple perspectives fairly and acknowledge the debate without taking "
            "a definitive side."
        ),
        "dietary_medical_advice": (
            "This question involves dietary restrictions or medical concerns. Provide "
            "relevant culinary information but clearly recommend consulting a medical "
            "professional or allergist for safety-critical decisions."
        ),
        "molecular_gastronomy": (
            "This question involves molecular gastronomy or modernist techniques "
            "outside classical French training. Acknowledge what you know from "
            "classical tradition and be honest about the limits of that knowledge."
        ),
        "proprietary_recipes": (
            "This question asks for a proprietary or secret recipe. Acknowledge that "
            "the exact recipe is not publicly available and offer what is known about "
            "the general technique or style, or suggest a publicly available approximation."
        ),
        "personal_taste": (
            "This question is a matter of personal taste. Present multiple valid "
            "perspectives, explain the reasoning behind different preferences, and "
            "avoid declaring one answer as objectively correct."
        ),
        "insufficient_information": (
            "This question lacks critical context needed for a useful answer. Ask "
            "clarifying questions and explain what information you'd need to give "
            "a proper answer."
        ),
    }

    guidance = category_guidance.get(category, "Respond appropriately.")

    system_prompt = (
        "You are a knowledgeable French cuisine expert who knows when to qualify, "
        "defer, or acknowledge uncertainty. You never give overconfident answers "
        "on topics that are genuinely uncertain, debatable, or outside your expertise. "
        "Write naturally as a chef would explain to a student — no excessive markdown "
        "or bullet points.\n\n"
        f"GUIDANCE FOR THIS RESPONSE: {guidance}"
    )

    return call_openrouter(
        api_key, SYNTHESIZER_MODEL, system_prompt, prompt, temperature=0.7
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DD v2 — Type 4 Boundary Case Generator"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=150,
        help="Target total boundary cases (default: 150)",
    )
    parser.add_argument(
        "--prompts-cache",
        default="dd-v2/data/prompts/chef_boundary_prompts.jsonl",
        help="Cache file for generated boundary prompts (avoids re-generating)",
    )
    args = parser.parse_args()

    api_key = get_api_key()

    # Resume support
    completed = load_completed_indices(args.output)
    print(f"Already completed: {len(completed)} boundary cases")

    # Step 1: Generate or load boundary scenario prompts
    all_scenarios = []

    if os.path.exists(args.prompts_cache):
        print(f"Loading cached boundary prompts from {args.prompts_cache}")
        with open(args.prompts_cache, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_scenarios.append(json.loads(line))
        print(f"Loaded {len(all_scenarios)} cached prompts")
    else:
        print("Generating boundary scenario prompts...")
        prompt_index = 0
        for category, config in CATEGORIES.items():
            questions = generate_scenario_prompts(api_key, category, config)
            for q in questions:
                all_scenarios.append({
                    "prompt": q,
                    "category": category,
                    "prompt_index": prompt_index,
                })
                prompt_index += 1
            time.sleep(RATE_LIMIT_BETWEEN_STAGES)

        # Cache the prompts
        os.makedirs(os.path.dirname(args.prompts_cache) or ".", exist_ok=True)
        with open(args.prompts_cache, "w", encoding="utf-8") as f:
            for sc in all_scenarios:
                f.write(json.dumps(sc, ensure_ascii=False) + "\n")
        print(f"Cached {len(all_scenarios)} boundary prompts to {args.prompts_cache}")

    # Step 2: Generate deferral responses for each scenario
    stats = {
        "total": len(all_scenarios),
        "skipped_already_done": 0,
        "responses_generated": 0,
        "deferral_verified": 0,
        "deferral_missing": 0,
        "api_errors": 0,
    }

    for scenario in all_scenarios:
        prompt_index = scenario.get("prompt_index", 0)
        prompt = scenario["prompt"]
        category = scenario["category"]

        if prompt_index in completed:
            stats["skipped_already_done"] += 1
            continue

        print(
            f"\n[{stats['responses_generated'] + stats['skipped_already_done'] + 1}"
            f"/{stats['total']}] ({category}) {prompt[:70]}..."
        )

        # Generate deferral response
        response = generate_deferral_response(api_key, prompt, category)
        time.sleep(RATE_LIMIT_BETWEEN_REQUESTS)

        if not response:
            stats["api_errors"] += 1
            print("  ERROR: Failed to generate response")
            continue

        stats["responses_generated"] += 1

        # Verify deferral language is present
        has_deferral, matched = check_deferral_language(response)

        if has_deferral:
            stats["deferral_verified"] += 1
            print(f"  Deferral verified ({len(matched)} pattern(s) matched)")
        else:
            stats["deferral_missing"] += 1
            print("  WARNING: No deferral language detected — saving anyway with flag")

        # Save result
        result = {
            "prompt_index": prompt_index,
            "type": "boundary_case",
            "category": category,
            "prompt": prompt,
            "response": response,
            "deferral_verified": has_deferral,
            "deferral_patterns_matched": len(matched),
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
        }
        save_result(result, args.output)

    # Print stats
    print(f"\n{'='*60}")
    print("Boundary Case Generation Complete")
    print(f"  Total scenarios:           {stats['total']}")
    print(f"  Skipped (already done):    {stats['skipped_already_done']}")
    print(f"  Responses generated:       {stats['responses_generated']}")
    print(f"  Deferral verified:         {stats['deferral_verified']}")
    print(f"  Deferral missing:          {stats['deferral_missing']}")
    print(f"  API errors:                {stats['api_errors']}")
    print(f"  Output: {args.output}")

    # Per-category breakdown
    if os.path.exists(args.output):
        cat_counts: dict[str, int] = {}
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    cat = obj.get("category", "unknown")
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print("\n  Per-category counts:")
        for cat, count in sorted(cat_counts.items()):
            target = CATEGORIES.get(cat, {}).get("count", "?")
            print(f"    {cat}: {count}/{target}")


if __name__ == "__main__":
    main()
