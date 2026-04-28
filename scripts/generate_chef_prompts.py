#!/usr/bin/env python3
"""
Generate 700 French cuisine prompts for the French Chef specialist model.

Uses Gemma 4 26B-A4B via OpenRouter to generate diverse, hard-verifiable
prompts across 7 categories and 3 difficulty levels.

Usage:
    python generate_chef_prompts.py                # Full run (700 prompts)
    python generate_chef_prompts.py --dry-run      # Test with 10 prompts
    python generate_chef_prompts.py --validate      # Validate existing prompts
"""

import argparse
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-4-26b-a4b-it"
BATCH_SIZE = 20
RATE_LIMIT_SECONDS = 1.0

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"
OUTPUT_FILE = OUTPUT_DIR / "chef_prompts.jsonl"

# Category definitions: name -> (target_count, description, example_patterns)
CATEGORIES = {
    "technique": {
        "target": 120,
        "desc": "French cooking technique explanations",
        "patterns": [
            "Explain the technique of X",
            "What is the difference between X and Y",
            "When would you use X instead of Y",
        ],
    },
    "regional_dish": {
        "target": 120,
        "desc": "Regional French dish identification and provenance",
        "patterns": [
            "What region is X from?",
            "Name traditional dishes from X region",
            "What makes X authentic?",
        ],
    },
    "mother_sauce": {
        "target": 100,
        "desc": "Mother sauces, derivatives, and sauce-making",
        "patterns": [
            "What are the base ingredients of X?",
            "How do you make X from Y?",
            "What's the difference between X and Y sauce?",
        ],
    },
    "ingredient_substitution": {
        "target": 100,
        "desc": "Ingredient roles and substitutions in French cooking",
        "patterns": [
            "Can I substitute X for Y in Z?",
            "What's the role of X in Y?",
            "Why does recipe X call for Y?",
        ],
    },
    "classical_preparation": {
        "target": 80,
        "desc": "Classical French preparations and mise en place",
        "patterns": [
            "How do you prepare X?",
            "What is a X and when is it used?",
            "Describe the components of X",
        ],
    },
    "temperature_timing": {
        "target": 80,
        "desc": "Temperature control and timing in French cooking",
        "patterns": [
            "What temperature for X?",
            "How do you know when X is done?",
            "What are the stages of Y?",
        ],
    },
    "error_correction": {
        "target": 100,
        "desc": "Diagnosing and fixing cooking mistakes",
        "patterns": [
            "I made X but it turned out Y, what went wrong?",
            "My sauce broke, how do I fix it?",
            "I overcooked X, can I salvage it?",
        ],
    },
}

TOTAL_TARGET = sum(c["target"] for c in CATEGORIES.values())  # 700

DIFFICULTY_DIST = {"easy": 0.30, "medium": 0.40, "hard": 0.30}

# Reference database of real French culinary terms for --validate
CULINARY_TERMS = {
    # Mother sauces
    "bechamel", "veloute", "espagnole", "hollandaise", "tomate",
    # Derivative sauces
    "mornay", "soubise", "nantua", "supreme", "allemande", "bercy",
    "bordelaise", "chasseur", "robert", "lyonnaise", "bearnaise",
    "choron", "foyot", "maltaise", "mousseline", "beurre blanc",
    "beurre rouge", "beurre noisette", "beurre noir",
    # Techniques
    "braise", "braiser", "sauter", "pocher", "poach", "blanch",
    "blanchir", "flamber", "flambe", "deglaze", "deglacer",
    "chiffonade", "julienne", "brunoise", "mirepoix", "bouquet garni",
    "roux", "liaison", "monter au beurre", "confit", "sous vide",
    "en croute", "gratiner", "gratin", "emulsion", "reduction",
    "clarification", "consomme", "demi-glace", "glace de viande",
    "fond", "fumet", "court-bouillon", "bain-marie", "tempering",
    "caramelize", "degorger", "larding", "barding", "truss",
    "chine", "tournee", "tourne", "paysanne", "ciseler",
    "concasser", "emincer", "escalope",
    # Knife cuts
    "batonnet", "allumette", "macedoine", "jardiniere",
    # Pastry / baking
    "pate brisee", "pate sucree", "pate sablee", "pate feuilletee",
    "pate a choux", "choux", "genoise", "biscuit", "meringue",
    "creme patissiere", "creme anglaise", "creme chantilly",
    "ganache", "praline", "nougatine", "tuile", "palmier",
    "mille-feuille", "paris-brest", "saint-honore", "tarte tatin",
    "clafoutis", "souffle", "crepe", "galette", "brioche",
    "croissant", "pain de campagne", "baguette",
    # Dishes
    "bouillabaisse", "cassoulet", "coq au vin", "boeuf bourguignon",
    "blanquette de veau", "pot-au-feu", "ratatouille", "quiche lorraine",
    "quiche", "gratin dauphinois", "soupe a l'oignon", "nicoise",
    "salade nicoise", "tapenade", "aioli", "pistou", "pissaladiere",
    "socca", "duck confit", "confit de canard", "magret",
    "foie gras", "terrine", "pate", "rillettes", "galantine",
    "charcuterie", "quenelle", "vol-au-vent", "croquembouche",
    "profiterole", "eclair", "macaron", "canele", "kouign-amann",
    "far breton", "flamiche", "tartiflette", "aligot",
    "fondue", "raclette", "choucroute", "baeckeoffe",
    "flammekueche", "tarte flambee", "navarin", "daube",
    # Regions
    "provence", "alsace", "burgundy", "bourgogne", "normandy",
    "normandie", "brittany", "bretagne", "lyon", "lyonnais",
    "bordeaux", "basque", "pays basque", "languedoc", "perigord",
    "champagne", "lorraine", "picardy", "ile-de-france", "savoie",
    "corsica", "corse", "gascon", "gascony",
    # Equipment / terms
    "mandoline", "chinois", "tamis", "cocotte", "sautoir",
    "sauteuse", "rondeaux", "russe", "bain-marie",
    "mise en place", "brigade de cuisine", "commis", "chef de partie",
    "sous chef", "garde manger", "poissonnier", "saucier",
    "rotisseur", "patissier", "entremetier", "tournant",
    # Temperatures / stages
    "soft ball", "hard ball", "soft crack", "hard crack",
    "nappe", "thread stage", "pearl stage",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing_prompts() -> list[dict]:
    """Load already-generated prompts from the output file."""
    prompts = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        prompts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return prompts


def count_by_category(prompts: list[dict]) -> dict[str, int]:
    """Return counts per category."""
    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for p in prompts:
        cat = p.get("category", "")
        if cat in counts:
            counts[cat] += 1
    return counts


def next_id(prompts: list[dict]) -> int:
    """Determine next numeric ID from existing prompts."""
    max_id = 0
    for p in prompts:
        pid = p.get("id", "")
        m = re.search(r"(\d+)$", pid)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def fuzzy_duplicate(new_prompt: str, existing: list[str], threshold: float = 0.80) -> bool:
    """Return True if new_prompt is >threshold similar to any existing prompt."""
    new_lower = new_prompt.lower().strip()
    for ex in existing:
        ratio = SequenceMatcher(None, new_lower, ex.lower().strip()).ratio()
        if ratio > threshold:
            return True
    return False


def pick_difficulty(index: int, total: int) -> str:
    """Deterministic difficulty assignment following the 30/40/30 distribution."""
    frac = index / max(total, 1)
    if frac < 0.30:
        return "easy"
    elif frac < 0.70:
        return "medium"
    else:
        return "hard"


def build_generation_prompt(category: str, info: dict, difficulty: str, count: int) -> str:
    """Build the system+user prompt for Gemma to generate prompts."""
    pattern_lines = "\n".join(f"  - {p}" for p in info["patterns"])
    return (
        f"You are an expert French cuisine curriculum designer. "
        f"Generate exactly {count} unique question prompts about: {info['desc']}.\n\n"
        f"Category: {category}\n"
        f"Difficulty: {difficulty}\n"
        f"  - easy = single-fact lookup (e.g., 'What are the five French mother sauces?')\n"
        f"  - medium = combining 2-3 facts (e.g., 'How does a veloute differ from a bechamel, "
        f"and when would you choose one over the other?')\n"
        f"  - hard = nuanced analysis or multi-step reasoning (e.g., 'A recipe calls for making "
        f"a sauce Nantua but you have no crayfish butter -- describe two alternative approaches "
        f"and explain how each changes the final dish.')\n\n"
        f"Example prompt patterns:\n{pattern_lines}\n\n"
        f"Rules:\n"
        f"- Every prompt must be answerable with hard-verifiable culinary facts.\n"
        f"- Reference REAL French culinary terms, dishes, regions, techniques, and ingredients.\n"
        f"- Vary phrasing and structure -- do not repeat the same template.\n"
        f"- Prompts should be questions or instructions a culinary student might ask.\n"
        f"- Do NOT include answers, only the prompts.\n\n"
        f"Return a JSON array of exactly {count} strings. No markdown fences, no commentary -- "
        f"just the JSON array."
    )


def call_gemma(prompt: str, api_key: str) -> list[str]:
    """Call Gemma 4 26B via OpenRouter and parse the response as a list of strings."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tinmanlabsl/Models",
        "X-Title": "Tinman Chef Prompt Generator",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 4096,
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Strip markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content.strip())

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract a JSON array from the response
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            print(f"  [WARN] Could not parse response as JSON. Raw content:\n{content[:500]}")
            return []

    if isinstance(parsed, list):
        return [str(item) for item in parsed if isinstance(item, str) or isinstance(item, dict)]
    return []


def validate_prompts(prompts: list[dict]) -> None:
    """Check that prompts reference real culinary terms from the reference DB."""
    terms_lower = {t.lower() for t in CULINARY_TERMS}
    no_match = []
    for p in prompts:
        text = p["prompt"].lower()
        found = any(term in text for term in terms_lower)
        if not found:
            no_match.append(p)

    total = len(prompts)
    matched = total - len(no_match)
    print(f"\nValidation results:")
    print(f"  Total prompts: {total}")
    print(f"  Referencing known culinary terms: {matched} ({100*matched/max(total,1):.1f}%)")
    print(f"  No recognized term found: {len(no_match)} ({100*len(no_match)/max(total,1):.1f}%)")

    if no_match:
        print(f"\nPrompts without recognized culinary terms (first 20):")
        for p in no_match[:20]:
            print(f"  [{p['id']}] ({p['category']}/{p['difficulty']}) {p['prompt'][:100]}")


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_prompts(dry_run: bool = False) -> None:
    """Main prompt generation loop with resume support."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing progress
    existing = load_existing_prompts()
    counts = count_by_category(existing)
    existing_texts = [p["prompt"] for p in existing]
    current_id = next_id(existing)

    total_existing = sum(counts.values())
    target = 10 if dry_run else TOTAL_TARGET
    print(f"Loaded {total_existing} existing prompts. Target: {target}.")

    if total_existing >= target:
        print("Already at target. Nothing to generate.")
        return

    # Build work plan: list of (category, difficulty, count) batches
    work: list[tuple[str, str, int]] = []
    for cat, info in CATEGORIES.items():
        cat_target = info["target"]
        if dry_run:
            # Scale down proportionally for dry run
            cat_target = max(1, round(info["target"] / TOTAL_TARGET * 10))
        remaining = cat_target - counts.get(cat, 0)
        if remaining <= 0:
            continue

        # Split remaining across difficulties
        easy_n = max(1, round(remaining * DIFFICULTY_DIST["easy"]))
        hard_n = max(1, round(remaining * DIFFICULTY_DIST["hard"]))
        medium_n = remaining - easy_n - hard_n
        if medium_n < 0:
            medium_n = 0
            easy_n = remaining // 2
            hard_n = remaining - easy_n

        for diff, n in [("easy", easy_n), ("medium", medium_n), ("hard", hard_n)]:
            if n <= 0:
                continue
            # Split into batches of BATCH_SIZE
            while n > 0:
                batch = min(n, BATCH_SIZE)
                work.append((cat, diff, batch))
                n -= batch

    print(f"Work plan: {len(work)} API calls to generate {sum(w[2] for w in work)} prompts.")

    generated_count = 0
    duplicates_skipped = 0

    for i, (cat, diff, batch_count) in enumerate(work):
        if total_existing + generated_count >= target:
            break

        info = CATEGORIES[cat]
        prompt = build_generation_prompt(cat, info, diff, batch_count)

        print(f"\n[{i+1}/{len(work)}] Requesting {batch_count} {diff} prompts for '{cat}'...")

        try:
            results = call_gemma(prompt, api_key)
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] API call failed: {e}")
            time.sleep(RATE_LIMIT_SECONDS * 3)
            continue
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  [ERROR] Response parsing failed: {e}")
            time.sleep(RATE_LIMIT_SECONDS)
            continue

        # Process results
        batch_added = 0
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for raw in results:
                # Handle case where Gemma returns dicts instead of strings
                if isinstance(raw, dict):
                    text = raw.get("prompt", raw.get("question", str(raw)))
                else:
                    text = str(raw).strip()

                if not text or len(text) < 10:
                    continue

                # Deduplicate
                if fuzzy_duplicate(text, existing_texts):
                    duplicates_skipped += 1
                    continue

                entry = {
                    "id": f"chef_{current_id:04d}",
                    "prompt": text,
                    "category": cat,
                    "difficulty": diff,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                existing_texts.append(text)
                current_id += 1
                batch_added += 1
                generated_count += 1

        print(f"  Added {batch_added}/{len(results)} prompts. "
              f"Total: {total_existing + generated_count}/{target}. "
              f"Dupes skipped: {duplicates_skipped}.")

        # Rate limiting
        time.sleep(RATE_LIMIT_SECONDS)

    print(f"\nDone. Generated {generated_count} new prompts. "
          f"Duplicates skipped: {duplicates_skipped}. "
          f"Total on disk: {total_existing + generated_count}.")
    print(f"Output: {OUTPUT_FILE}")

    # Print summary by category
    final = load_existing_prompts()
    final_counts = count_by_category(final)
    print(f"\nCategory breakdown:")
    for cat, info in CATEGORIES.items():
        actual = final_counts.get(cat, 0)
        target_n = info["target"]
        status = "OK" if actual >= target_n else f"NEED {target_n - actual} MORE"
        print(f"  {cat:25s}: {actual:4d}/{target_n:4d}  {status}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate French cuisine prompts for DD training pipeline."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate only 10 prompts (test mode)."
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing prompts against culinary term reference DB."
    )
    args = parser.parse_args()

    if args.validate:
        prompts = load_existing_prompts()
        if not prompts:
            print(f"No prompts found at {OUTPUT_FILE}")
            sys.exit(1)
        validate_prompts(prompts)
    else:
        generate_prompts(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
