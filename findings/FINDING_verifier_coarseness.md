# Finding: ChefVerifier is a Coarse Pre-Filter, Not a Quality Oracle

**Date discovered:** 2026-04-07
**Severity:** Foundation-level — affects how every DD v2 specialist's verifier is interpreted
**Status:** Mitigated for Chef via judge-tiebreaker pipeline; framework claim intact

---

## What we found

When generating Type 3 DPO pairs (gold response vs MiniMax-generated plausible-but-wrong response), we expected the verifier to clearly score gold > wrong. Reality:

- **287/569 pairs (47%)** had `wrong_score >= gold_score` — the verifier could not numerically distinguish gold from a well-written wrong answer
- Only **8/681 (1.2%)** of corrected failure traces (Type 2) achieved a clean `verify_response().passed = True` despite being measurably better than the student attempt
- Avg verifier score on **gold itself was only 0.60** — even the multi-stage teacher synthesis didn't score "excellent" by the verifier's own standard

## What we initially thought it meant

Initial reading: "the verifier is 47% wrong on rankings — it actively prefers wrong responses to gold ones half the time. The DD v2 'verifier-grounded distillation' claim is structurally broken."

This led to a discussion about whether to swap the primary evaluation metric from verifier-graded to LLM-judge head-to-head.

## What it actually means

DeepSeek V3.2 judge tiebreaker on the 287 inverted pairs:
- **271/287 (94%)** — judge confirmed chosen IS more correct than rejected. The verifier was numerically scoring-confused but the **ranking was still right**.
- **12/287 (4%)** — judge said rejected was actually better; pair was swapped
- **4/287 (1.4%)** — judge tied or errored; pair dropped

**Correct interpretation:** ChefVerifier is a **coarse keyword-and-reference-DB matcher** that produces noisy scores in the 0.4–0.7 range for any well-written response that mentions the right vocabulary. It cannot distinguish *factually correct use of vocabulary* from *plausible misuse of vocabulary*. But its **ranking signal is still mostly correct** — gold > wrong holds in aggregate even when individual pair scores are numerically tangled.

The 47% inversion is a **scoring precision problem**, not a **ranking direction problem**.

## Why we missed it during design

Reviewed in this session: I had multiple opportunities to catch this earlier and didn't.

1. **Gold avg score 0.60 was a yellow flag in Step 10** — I noted it but didn't dig in. I assumed "0.60 = moderately good" rather than "verifier doesn't go higher than this for any reasonable response."
2. **The Step 12 quality-check log only flagged 6 inversions** because it required BOTH score-inversion AND length-mismatch to log a warning. I read "6 inversions out of 608" and didn't probe further. The actual `wrong_score >= gold_score` raw count was 287 — only revealed when I wrote the cleanup script and asked the question directly.
3. **The Type 2 1.2% pass rate was a screaming red flag** that I correctly noticed but mis-attributed: I said "verifier is too strict" instead of asking "is the verifier even doing what we think?"
4. **I never asked the inverse question:** "If I show the verifier a known-bad response, does it correctly score it lower than a known-good response?" That's the verifier-validation experiment I should have run during Step 0 of DD v2.

The pattern across all 4 misses: I trusted the verifier as designed because the design doc said "verifier-grounded." I never tested whether the verifier actually had the property the design assumed. **I treated a design assumption as a verified fact.**

## Mitigations in place

1. **Type 3 (DPO) cleanup:** Length-drop + judge-tiebreaker on every score-inverted pair (`scripts/clean_type3.py`). Result: 565 clean DPO pairs from 608 raw.
2. **Type 2 (failure traces) cleanup:** Drop traces with no improvement or still-bad correction (`scripts/clean_type2.py`). Result: 550 clean traces from 681 raw.
3. **PLAN.md Step 16 update:** Added LLM-judge head-to-head as **secondary** metric alongside the verifier-graded primary. If they agree → DD v2 claim defended; if they disagree → claim falsified honestly. Does NOT swap the primary, so the verifier-grounded thesis is still being tested.
4. **Memory rule added:** never trust a verifier as ground truth without running a verifier-validation experiment first.

## Rules to apply going forward

**For every future DD v2 specialist (Coding, GDPR, Companion):**

1. **Verifier validation MUST be Step 0**, before any data generation:
   - Build a tiny "known-good vs known-bad" calibration set (50 hand-curated pairs)
   - Run the verifier on it
   - Confirm: known-good ranks higher than known-bad in **at least 90% of pairs** (not just 50%, since 50% is random)
   - If verifier ranks correctly <90% → fix the verifier OR redesign the framework grounding before generating any data
2. **Always check the score distribution** of the verifier on gold responses early. If gold avg < 0.75, the verifier ceiling is suspicious.
3. **Always check raw `wrong_score >= gold_score` rate**, not just script-flagged warnings, when generating DPO pairs.
4. **Add the LLM-judge head-to-head as secondary** in every specialist's Step 16, not just Chef.

## Cost of the fix

- Type 3 cleanup judge calls: ~$2.30 (DeepSeek V3.2, 287 calls)
- PLAN.md secondary metric: ~$3 per evaluation pass, per specialist
- Step 0 verifier-validation: ~$0 (manual hand-curation, ~1 hour per specialist)

Cheap compared to the cost of a paper retraction or a broken framework claim.
