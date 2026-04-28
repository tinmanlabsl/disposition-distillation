"""
Chef Verifier — verify model culinary responses against a reference DB
of hard-verifiable French cuisine facts.

Usage:
    from chef_verifier import ChefVerifier
    v = ChefVerifier()  # loads default DB path
    result = v.verify_response(response_text, prompt_text)
    score  = v.score_response(response_text, prompt_text)
"""

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    """Remove diacritics so 'Béchamel' matches 'bechamel'."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    return re.sub(r"\s+", " ", _strip_accents(s).lower().strip())


def _fuzzy_match(candidate: str, targets: list[str], threshold: float = 0.80) -> Optional[str]:
    """Return the best-matching target if similarity >= threshold, else None."""
    c_norm = _normalize(candidate)
    best, best_ratio = None, 0.0
    for t in targets:
        t_norm = _normalize(t)
        # fast exact check
        if c_norm == t_norm:
            return t
        ratio = SequenceMatcher(None, c_norm, t_norm).ratio()
        if ratio > best_ratio:
            best, best_ratio = t, ratio
    return best if best_ratio >= threshold else None


def _extract_quoted_and_capitalized(text: str) -> list[str]:
    """Pull out quoted phrases and capitalized multi-word sequences as candidates."""
    candidates = set()
    # quoted phrases
    for m in re.finditer(r'["\u201c\u201d]([^"\u201c\u201d]{2,60})["\u201c\u201d]', text):
        candidates.add(m.group(1).strip())
    # Capitalized sequences (e.g. "Sauce Béchamel", "Brunoise")
    for m in re.finditer(r'\b([A-Z\u00C0-\u024F][a-z\u00E0-\u024F]+(?:\s+[A-Za-z\u00C0-\u024F][a-z\u00E0-\u024F]+)*)\b', text):
        candidates.add(m.group(1).strip())
    return list(candidates)


def _extract_temperatures(text: str) -> list[dict]:
    """Extract temperature claims like '180°C', '350 F', '200 degrees'."""
    temps = []
    # patterns: 180°C, 180 °C, 180C, 180 degrees C/F, 350F, 350 °F
    for m in re.finditer(
        r'(\d{2,4})\s*(?:°|degrees?\s*)?\s*([CcFf])\b', text
    ):
        value = int(m.group(1))
        unit = m.group(2).upper()
        temps.append({"value": value, "unit": unit, "text": m.group(0)})
    return temps


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

class ChefVerifier:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.environ.get(
                "CHEF_REF_DB",
                str(Path(__file__).resolve().parent / ".." / "data" / "french_cuisine_reference.json"),
            )
        with open(db_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)

        # Build lookup indexes (normalized name -> original key/entry)
        self._build_indexes()

    # ---- index building ----------------------------------------------------

    def _build_indexes(self):
        """Pre-compute normalized name lookups for every DB category."""
        self.all_names: dict[str, str] = {}  # norm_name -> original_name
        self.name_to_category: dict[str, str] = {}  # norm_name -> category

        # mother_sauces: list of {name, base, key_ingredients, ...}
        self.mother_sauces = {}
        for s in self.db.get("mother_sauces", []):
            name = s.get("name", "")
            norm = _normalize(name)
            self.mother_sauces[norm] = s
            self.all_names[norm] = name
            self.name_to_category[norm] = "mother_sauces"

        # knife_cuts: list of {name, definition/description, dimensions, ...}
        self.knife_cuts = {}
        for k in self.db.get("knife_cuts", []):
            name = k.get("name", "")
            norm = _normalize(name)
            self.knife_cuts[norm] = k
            self.all_names[norm] = name
            self.name_to_category[norm] = "knife_cuts"

        # regional_dishes: list of {name, region, ...}
        self.regional_dishes = {}
        for d in self.db.get("regional_dishes", []):
            name = d.get("name", "")
            norm = _normalize(name)
            self.regional_dishes[norm] = d
            self.all_names[norm] = name
            self.name_to_category[norm] = "regional_dishes"

        # classical_techniques: list of {name, definition/description, ...}
        self.techniques = {}
        for t in self.db.get("classical_techniques", []):
            name = t.get("name", "")
            norm = _normalize(name)
            self.techniques[norm] = t
            self.all_names[norm] = name
            self.name_to_category[norm] = "classical_techniques"

        # temperatures: may be dict-of-lists or flat list
        self.temperatures = {}
        temps_raw = self.db.get("temperatures", [])
        if isinstance(temps_raw, dict):
            # Flatten: {"sugar_stages": [...], "oven_temps": [...], ...}
            temp_items = []
            for group in temps_raw.values():
                if isinstance(group, list):
                    temp_items.extend(group)
        elif isinstance(temps_raw, list):
            temp_items = temps_raw
        else:
            temp_items = []
        for t in temp_items:
            if not isinstance(t, dict):
                continue
            key = t.get("context") or t.get("name") or ""
            norm = _normalize(key)
            self.temperatures[norm] = t
            self.all_names[norm] = key
            self.name_to_category[norm] = "temperatures"

        # classical_preparations: list of {name, ingredients, ...}
        self.preparations = {}
        for p in self.db.get("classical_preparations", []):
            name = p.get("name", "")
            norm = _normalize(name)
            self.preparations[norm] = p
            self.all_names[norm] = name
            self.name_to_category[norm] = "classical_preparations"

        self._all_name_list = list(self.all_names.values())

    # ---- extraction helper -------------------------------------------------

    def _find_mentions(self, text: str) -> list[tuple[str, Optional[str]]]:
        """Return list of (candidate_text, matched_db_name_or_None)."""
        candidates = _extract_quoted_and_capitalized(text)
        # Also try matching each known DB name directly in the text
        text_norm = _normalize(text)
        for norm_name, orig_name in self.all_names.items():
            if norm_name in text_norm and orig_name not in candidates:
                candidates.append(orig_name)

        results = []
        seen = set()
        for c in candidates:
            c_norm = _normalize(c)
            if c_norm in seen or len(c_norm) < 3:
                continue
            seen.add(c_norm)
            match = _fuzzy_match(c, self._all_name_list, threshold=0.80)
            results.append((c, match))
        return results

    # ---- public verification functions -------------------------------------

    def check_technique_accuracy(self, response: str) -> dict:
        """Check technique names and definitions against DB."""
        mentioned, correct, incorrect, unknown = [], [], [], []
        text_norm = _normalize(response)

        for candidate, match in self._find_mentions(response):
            if match is None:
                continue
            match_norm = _normalize(match)
            if match_norm not in self.techniques:
                continue
            mentioned.append(candidate)
            db_entry = self.techniques[match_norm]
            desc = db_entry.get("definition") or db_entry.get("description") or ""
            if not desc:
                unknown.append(candidate)
                continue
            # Check if response contains words consistent with the definition
            # (lightweight: just check the candidate is mentioned; detailed
            # definition checking would need NLI which is out of scope)
            desc_keywords = set(_normalize(desc).split()) - {"a", "an", "the", "of", "to", "is", "in", "and", "or", "with"}
            resp_words = set(text_norm.split())
            overlap = desc_keywords & resp_words
            if len(overlap) >= min(3, len(desc_keywords) // 2):
                correct.append(candidate)
            else:
                # Can't confirm correctness — mark unknown rather than incorrect
                unknown.append(candidate)

        return {"mentioned": mentioned, "correct": correct, "incorrect": incorrect, "unknown": unknown}

    def check_regional_attribution(self, response: str) -> dict:
        """Check dish-region claims against DB."""
        mentioned, correct, incorrect, unknown = [], [], [], []

        # Build region keyword sets for matching
        region_keywords = {}
        for norm, entry in self.regional_dishes.items():
            region = entry.get("region", "")
            region_keywords[norm] = _normalize(region)

        text_norm = _normalize(response)

        for candidate, match in self._find_mentions(response):
            if match is None:
                continue
            match_norm = _normalize(match)
            if match_norm not in self.regional_dishes:
                continue
            mentioned.append(candidate)
            expected_region = region_keywords.get(match_norm, "")
            if not expected_region:
                unknown.append(candidate)
                continue
            # Check if the expected region appears near the dish mention in text
            if expected_region in text_norm:
                correct.append(candidate)
            else:
                # Region not mentioned — might be correct but unverifiable
                unknown.append(candidate)

        return {"mentioned": mentioned, "correct": correct, "incorrect": incorrect, "unknown": unknown}

    def check_ingredient_correctness(self, response: str) -> dict:
        """For mother sauces / classical preparations, check ingredient claims."""
        mentioned, correct, incorrect, unknown = [], [], [], []
        text_norm = _normalize(response)

        # Combine mother sauces and preparations
        ingredient_sources = {}
        for norm, entry in self.mother_sauces.items():
            ingredients = entry.get("key_ingredients") or entry.get("ingredients") or []
            if isinstance(ingredients, str):
                ingredients = [i.strip() for i in ingredients.split(",")]
            ingredient_sources[norm] = [_normalize(i) for i in ingredients]
        for norm, entry in self.preparations.items():
            ingredients = entry.get("ingredients") or entry.get("key_ingredients") or []
            if isinstance(ingredients, str):
                ingredients = [i.strip() for i in ingredients.split(",")]
            ingredient_sources[norm] = [_normalize(i) for i in ingredients]

        for candidate, match in self._find_mentions(response):
            if match is None:
                continue
            match_norm = _normalize(match)
            if match_norm not in ingredient_sources:
                continue
            mentioned.append(candidate)
            db_ingredients = ingredient_sources[match_norm]
            if not db_ingredients:
                unknown.append(candidate)
                continue
            # Check how many DB ingredients appear in the response
            found = sum(1 for ing in db_ingredients if ing in text_norm)
            if found >= max(1, len(db_ingredients) // 2):
                correct.append(candidate)
            elif found == 0:
                # No ingredient overlap — could be incorrect or just not discussed
                unknown.append(candidate)
            else:
                unknown.append(candidate)

        return {"mentioned": mentioned, "correct": correct, "incorrect": incorrect, "unknown": unknown}

    def check_invented_references(self, response: str) -> dict:
        """Flag terms that look culinary but don't appear in our DB."""
        known, potentially_invented = [], []

        for candidate, match in self._find_mentions(response):
            if match is not None:
                known.append(candidate)
            else:
                # Only flag things that look culinary (heuristic: French-ish words,
                # food-related context). Skip generic English words.
                c_norm = _normalize(candidate)
                # Skip very short or very common words
                if len(c_norm) < 4:
                    continue
                # Simple heuristic: if it contains culinary-adjacent patterns
                culinary_hints = [
                    "sauce", "soupe", "creme", "pate", "tarte", "gratin",
                    "confit", "ragu", "puree", "mousse", "fond", "jus",
                    "braise", "flambe", "saute", "blanch", "poach", "roast",
                    "julienne", "brunoise", "chiffonade", "mirepoix",
                ]
                if any(h in c_norm for h in culinary_hints):
                    potentially_invented.append(candidate)
                # Also flag if it looks French (ends in common French suffixes)
                elif re.search(r'(aise|oise|ette|ade|ine|eau|aux|tion)\b', c_norm):
                    potentially_invented.append(candidate)

        return {"known": known, "potentially_invented": potentially_invented}

    def check_temperature_plausibility(self, response: str) -> dict:
        """Check temperature claims against DB ranges."""
        mentioned, plausible, implausible = [], [], []

        temps_found = _extract_temperatures(response)
        if not temps_found:
            return {"mentioned": [], "plausible": [], "implausible": []}

        # Collect all DB temperature ranges (convert to Celsius for comparison)
        db_ranges = []
        for entry in self.temperatures.values():
            # Handle multiple data formats
            tc = entry.get("temp_celsius")
            tcr = entry.get("temp_celsius_range")
            min_c = entry.get("min_c") or entry.get("min")
            max_c = entry.get("max_c") or entry.get("max")
            if tc is not None:
                db_ranges.append((float(tc), float(tc)))
            elif tcr is not None:
                parts = str(tcr).split("-")
                if len(parts) == 2:
                    db_ranges.append((float(parts[0]), float(parts[1])))
            elif min_c is not None and max_c is not None:
                db_ranges.append((float(min_c), float(max_c)))

        # If no DB ranges, everything is unknown
        if not db_ranges:
            return {
                "mentioned": [t["text"] for t in temps_found],
                "plausible": [],
                "implausible": [],
            }

        # Global plausible range: union of all DB ranges with some margin
        global_min = min(r[0] for r in db_ranges) - 20
        global_max = max(r[1] for r in db_ranges) + 50

        for t in temps_found:
            mentioned.append(t["text"])
            value_c = t["value"]
            if t["unit"] == "F":
                value_c = (t["value"] - 32) * 5 / 9

            if global_min <= value_c <= global_max:
                plausible.append(t["text"])
            else:
                implausible.append(t["text"])

        return {"mentioned": mentioned, "plausible": plausible, "implausible": implausible}

    def verify_response(self, response: str, prompt: str = "") -> dict:
        """Run all checks and return aggregate result."""
        checks = {
            "technique_accuracy": self.check_technique_accuracy(response),
            "regional_attribution": self.check_regional_attribution(response),
            "ingredient_correctness": self.check_ingredient_correctness(response),
            "invented_references": self.check_invented_references(response),
            "temperature_plausibility": self.check_temperature_plausibility(response),
        }

        # Determine overall pass/fail
        issues = []

        for name, result in checks.items():
            if name == "invented_references":
                if result["potentially_invented"]:
                    issues.append(f"{name}: {len(result['potentially_invented'])} potentially invented terms")
            elif name == "temperature_plausibility":
                if result["implausible"]:
                    issues.append(f"{name}: {len(result['implausible'])} implausible temperatures")
            else:
                if result["incorrect"]:
                    issues.append(f"{name}: {len(result['incorrect'])} incorrect claims")

        passed = len(issues) == 0

        return {
            "passed": passed,
            "issues": issues,
            "checks": checks,
            "prompt": prompt[:200] if prompt else "",
            "response_length": len(response),
        }

    def score_response(self, response: str, prompt: str = "") -> float:
        """Return 0.0-1.0 quality score based on verification checks.

        For each check: score = (correct + unknown*0.5) / total_mentioned.
        Final score = average across checks that had any mentions.
        """
        result = self.verify_response(response, prompt)
        scores = []

        for name, check in result["checks"].items():
            if name == "invented_references":
                # Score: known / (known + invented)
                total = len(check["known"]) + len(check["potentially_invented"])
                if total > 0:
                    scores.append(len(check["known"]) / total)
                continue

            if name == "temperature_plausibility":
                total = len(check["mentioned"])
                if total > 0:
                    scores.append(len(check["plausible"]) / total)
                continue

            # Standard checks: technique, regional, ingredient
            total = len(check["mentioned"])
            if total == 0:
                continue
            n_correct = len(check["correct"])
            n_unknown = len(check["unknown"])
            scores.append((n_correct + n_unknown * 0.5) / total)

        if not scores:
            return 0.5  # no verifiable claims found — neutral score

        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chef_verifier.py <response_text_or_file>")
        sys.exit(1)

    arg = sys.argv[1]
    if os.path.isfile(arg):
        with open(arg, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = arg

    v = ChefVerifier()
    result = v.verify_response(text)
    score = v.score_response(text)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nScore: {score:.3f}")
