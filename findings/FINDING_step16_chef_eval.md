# Finding: Step 16 Chef Eval — Verifier vs Judge vs Disposition

**Date:** 2026-04-08
**Status:** Step 16 + 16b complete; 16c/MCAS rubric in progress
**Severity:** Determines whether DD v2 Chef ships, pivots, or is reframed

---

## Summary

Verifier and judge BOTH say full_ddv2 is slightly worse than baseline on factual correctness for Chef. But sample inspection shows the gap is partly an artifact of (a) verifier coarseness/flakiness, (b) verifier failing to parse LaTeX-formatted answers, and (c) trained models giving more pedagogically framed answers that the verifier doesn't reward. The disposition axes (rubric judge + Chef-MCAS) are the actual test of the DD v2 claim and were still running at time of writing.

## Step 16 Primary (ChefVerifier, n=100 each, locked GEN_PARAMS)

| condition | avg_score | pass_rate | technique_clean | regional_clean | ingredient_clean | invented_zero |
|---|---|---|---|---|---|---|
| baseline | 0.279 | 0.02 | 1.000 | 1.000 | 1.000 | — |
| single_teacher_sft | 0.291 | 0.06 | — | — | — | — |
| multi_teacher_sft | 0.285 | 0.02 | — | — | — | — |
| sft_only | 0.270 | 0.04 | 1.000 | 1.000 | 1.000 | 0.040 |
| **full_ddv2** | **0.241** | **0.05** | 1.000 | 1.000 | 1.000 | 0.050 |

All 5 conditions cluster in 0.24–0.29. Per-prompt verifier variance is ~0.1–0.15, so the n=100 95% CI is ±0.02–0.03 — the conditions are inside each other's noise band.

## Step 16b Secondary (DeepSeek V3.2 head-to-head, full_ddv2 vs baseline)

- **full_ddv2 win-rate: 45%**
- **baseline win-rate: 52%**
- ties: 3%, errors: 0
- Judge AGREES with verifier direction (baseline edges out full_ddv2) but the gap is smaller (~7 points) than the verifier suggested (~14 points).
- Locked params: `temperature=0.0`, `max_tokens=10`, A/B order swap debias on alternate prompts.

## Sample inspection — three "worst" full_ddv2 prompts

Loaded local JSONs and compared `min(baseline,sft) - full_ddv2` deltas; pulled the top 3.

### Prompt 63 — limp pommes frites
- Verifier: baseline=1.00, sft_only=0.50, full_ddv2=**0.00**
- Reality: sft_only and full_ddv2 produce **near-identical responses** (same "I need a little more information" opener, same #1 cause, same 🍟 emoji, same structure). The verifier's 0.50 vs 0.00 split on near-identical text is **pure noise**, not a quality signal.

### Prompt 39 — soggy fries / oil temperature analysis
- Verifier: baseline=0.60, sft_only=0.50, full_ddv2=**0.00**
- Cause: full_ddv2 wrote temperatures as `$175^\circ\text{C}$` and `$350^\circ\text{F}$` (LaTeX formatting). The verifier almost certainly regex-matches plain text like `175°C` / `350°F` and **fails to recognize the LaTeX form**, counting the response as "missing temperature reference."
- This is a **verifier extraction bug** induced by a stylistic shift the training caused. DD nudged the model toward more "academic" formatting; the verifier punishes it.

### Prompt 29 — soufflé collapse
- Verifier: baseline=0.50, sft_only=0.50, full_ddv2=**0.00**
- full_ddv2 says "Premature Oven Opening (Thermal Shock)" as the #1 cause — that's the **canonical correct answer** in classical French technique. Baseline says "Under-Baking" which is less commonly cited. **full_ddv2 gives the more correct answer and is scored lower.**

## Pattern observed across the three samples

Trained models (sft_only, full_ddv2) share a distinct voice the baseline doesn't have:
- Hedging openers ("To give you the most accurate advice, I need a little more information")
- Pedagogical framing ("mise en place", "the physics is simple", defining terms)
- Acknowledging subjectivity / multiple valid approaches
- Asking clarifying questions when prompts are under-specified

These are **dispositional traits** — humble, deliberate, curious. Exactly what DD targets. **The verifier rewards none of this.** The 0.241 number does not measure what DD v2 was trying to install.

## Why the verifier is wrong here

ChefVerifier was designed (and previously found, see `FINDING_verifier_coarseness.md`) as a coarse keyword-and-reference-DB matcher. Confirmed extension of that finding:
1. **Stylistic regression**: when training shifts output formatting (LaTeX, markdown headers, more discursive prose), the verifier's regex-based extractors silently fail and score 0.
2. **Position bias**: when the canonical correct answer is listed second or framed differently, the verifier may not rank it.
3. **Pedagogical penalty**: more framing = more text the verifier doesn't recognize = lower score.

## What this means for DD v2

- The "verifier-grounded distillation" thesis from the DD v2 plan is **partially falsified for Chef**: training against the verifier did not raise the verifier score. The training data taught a recognizably different *style*, not raw fact density.
- Whether DD v2 actually *helps* depends on the disposition axes (Step 16c rubric judge, in progress). If the rubric shows lift on hedging/pedagogical/self-verification, DD v2 succeeded at its actual job and the verifier was the wrong oracle to optimize against.
- If the rubric also shows no lift, DD v2 trained a stylistic mask without behavioral content — and we pivot to Coding specialist (per PLAN's "If It Fails" branch).

## Conditions tracked + reproducibility

All 5 condition results saved to `dd-v2/eval/results/eval_<condition>.json` (on pod and locally). Each contains:
- `summary` block with all aggregate metrics
- `rows` array with prompt, response, score, per-axis breakdown
- (Step 16b) `judge_head2head.json` with verdict per prompt + locked judge params
- (Step 16c) `judge_rubric.json` (pending) with 5-axis disposition scores per response per condition

GEN_PARAMS (locked across all 5 model conditions):
- temperature=0.0 (do_sample=False, greedy)
- max_new_tokens=512, max_input_length=1024
- batch_size=32, padding_side='left'
- model: unsloth/gemma-4-E2B-it base + LoRA adapters, load_in_4bit=True
- prompts: type1_gold.jsonl[-100:] (held-out tail)
- verifier: ChefVerifier with REF_DB french_cuisine_reference.json

JUDGE_PARAMS (locked for 16b + 16c):
- judge_model: deepseek/deepseek-v3.2-exp via OpenRouter
- temperature=0.0, max_tokens=10 (16b) / 400 (16c)
- A/B swap debias on alternate prompts (16b)
- concurrency=5

## Step 16c — Rubric judge results (DeepSeek V3.2, 5 axes, n=100 each)

| condition | factual | hedging | pedagogical | self_verif | complete | **MCAS proxy** |
|---|---|---|---|---|---|---|
| baseline | **3.02** | **2.75** | 4.47 | **2.15** | **2.68** | **4.40** |
| sft_only | 2.69 | 2.60 | 4.46 | 2.06 | 2.64 | 2.50 |
| **full_ddv2** | **2.46** | **2.29** | 4.24 | **1.91** | **2.44** | **2.50** |

**MCAS proxy** = `hedging_appropriateness` averaged over the subset of prompts the judge classified as UNCERTAIN or AMBIGUOUS. This is the closest analog to the original DD claim — does the model hedge correctly when hedging is warranted?

**Key result:** baseline beats both trained models on **every single axis**. The MCAS proxy drops from **4.40 → 2.50** going from baseline to either trained model — a 1.90-point regression on a 5-point scale on the metric DD was specifically designed to improve. Pedagogical framing is the only axis where all three are roughly tied (4.24–4.47, near ceiling for the rubric).

## Diagnosis: stylistic mask, behavioral regression

Cross-referenced rubric scores against the head-to-head loss samples:
- Trained models reliably reproduce a *teacherly opener* ("It sounds like you're running into...", "I need a little more information"), markdown headers, and emoji garnish.
- They lose factual sharpness (concrete temperatures, canonical attributions) AND hedge in the wrong places.
- The Hollandaise re-emulsify case is the canonical failure: user asks for the *quickest fix*; full_ddv2 deflects into "since you can't easily fix the existing broken emulsion, you need to introduce a new, powerful emulsifier" — a stalling, vague non-answer.

DD v2 trained the *appearance* of disposition (the voice) without grounding it to the *appropriate context* (when to hedge vs when to act). On direct-action questions, the disposition becomes deflection.

## Architectural gap: no register grounding in the pipeline

Reviewed the 4-stage pipeline (Eager → Deliberate → Adversarial → Synthesizer). **None of the stages check whether the disposition is appropriate to the question type.** Every prompt is treated identically — ACTION, ANALYSIS, COMPARISON, EXPLANATION all get the same uniform "deliberate + humble" treatment.

Three concrete fixes needed for DD v2.1:

1. **Question-type tagging** — every training prompt tagged ACTION / ANALYSIS / COMPARISON / EXPLANATION / OPEN-ENDED. ACTION → direct answer first, framing optional. ANALYSIS → framing then conclusion. The model learns register-aware disposition, not uniform disposition.

2. **Adversarial deflection check** — MiniMax M2.7 stage prompt extended with: "If the user asked for an action and the response gave framing instead of action, mark FAIL." This catches the Hollandaise pattern at training time, before SFT.

3. **Synthesizer register matching** — Qwen synthesizer prompt extended with: "Match the register of the question. Direct asks get direct answers. Analytical asks get analytical answers. Disposition is a flavor, not a substitute for answering."

## Step 16d — Inference-time mitigation result

| run | verifier_score | pass_rate | invented |
|---|---|---|---|
| baseline | 0.279 | 0.02 | 5.79 |
| full_ddv2 | 0.241 | 0.05 | 5.00 |
| **full_ddv2 + sysprompt** | **0.282** | **0.14** | **3.84** |

Verifier metrics recovered above baseline. 2.8× pass_rate vs baseline. Invented references dropped 23%.

**But the rubric judge tells the deeper story:**

| condition | factual | hedging | pedagogical | self_verif | complete | MCAS |
|---|---|---|---|---|---|---|
| baseline | 3.02 | 2.75 | 4.47 | 2.15 | 2.68 | 4.40 |
| full_ddv2 | 2.46 | 2.29 | 4.24 | 1.91 | 2.44 | 2.50 |
| full_ddv2 + sysprompt | 2.49 | 2.32 | **3.80** | 1.98 | 2.61 | **0.00** |

The disposition axes barely moved. Pedagogical framing actually *dropped* (4.24 → 3.80) — the system prompt suppressed it. **MCAS = 0.00 is the headline:** with the sysprompt active, the judge classified **zero** of the 100 prompts as UNCERTAIN/AMBIGUOUS, meaning the model became uniformly direct. The MCAS subset is empty.

**Interpretation:** the system prompt traded over-hedging for over-directness. It did not restore *adaptive* disposition; it suppressed disposition entirely. Verifier improved because more concrete answers = more keyword hits, but the actual DD claim (context-aware hedging) is not satisfied in either state — neither the trained-without-prompt nor the trained-with-prompt model produces adaptive disposition.

## Methodological conclusion

DD v2 as currently designed cannot produce *adaptive* disposition. The pipeline installs *uniform* disposition (always hedge, always frame) and an inference-time switch can only turn it ON or OFF globally. The two operating points are:

- **No system prompt:** uniform high disposition → over-hedges direct asks → verifier and judge both penalize
- **System prompt:** uniform suppression of disposition → recovers verifier but kills the disposition signal → MCAS subset becomes empty

Neither matches the original DD claim of *behavioral tendency that activates when warranted*.

The structural fix is question-type tagging in the training data + register-aware pipeline prompts (v2.1, see "Architectural gap" section above). v2.1 would teach the model *when* to hedge, not just *to* hedge.

## Final decision tree

| Option | Cost | What it validates |
|---|---|---|
| (A) v2.1 Chef retrain | ~$10 + ~4h GPU | Whether the methodology can produce adaptive disposition after the pipeline fix |
| (B) Pivot to Coding | ~$15 + ~6h GPU | Whether the methodology validates on a domain where register-mismatch can't bite (action-by-default) |
| (C) Both, Coding first | ~$25 + ~10h GPU | Strongest validation: positive on Coding, then v2.1 Chef proves the fix generalizes |

This is for **method validation, not shipping**. The chef bot is not the deliverable — the disposition transfer claim is. (A) is the cleanest test of the v2.1 fix, (B) is the safest path to a positive result, (C) is the most rigorous.

## Step 16d — (legacy heading, original placeholder, see above)

Re-running full_ddv2 with a register-grounding system prompt to test whether the failure is generation-time or learned-in-weights:

> "Answer the user's question directly and concretely. If they ask for a fix, give the fix first, then explain. If they ask 'what is X', define X first. Use hedging or 'I need more information' ONLY when there is genuine ambiguity in the question. Do not deflect direct asks into framing."

Same 100 held-out prompts, same GEN_PARAMS, only addition is the system message. If verifier and rubric scores recover toward baseline → fix is cheap, no re-train. If they don't → over-hedging is baked into weights and only the v2.1 re-train will fix it.

Result: pending (eval running on pod).

## Decision

DD v2 Chef thesis as-trained is **falsified** on disposition axes. The pivot options are:

- **(A) Quick fix:** if Step 16d sysprompt mitigation closes the gap, ship Chef with a system prompt and document the gap honestly.
- **(B) Re-train v2.1:** add the three pipeline fixes above, regenerate Type-1 data with question-type tags, retrain SFT + DPO. ~1 day, ~$10.
- **(C) Pivot to Coding:** per PLAN's "If It Fails" branch — Coding has hard execution verification, much stronger signal than ChefVerifier, less prone to register mismatch (code is action by default).

Recommendation pending Step 16d result.

## Step 16f — Gold-anchored deterministic checklist judge (v3)

To eliminate length-bias and style-halo from the rubric judge, built a binary per-claim checklist: 100 hand-drafted gold checklists (3-5 required canonical claims each, e.g. for Hollandaise re-emulsify: "fresh yolk", "warm water", "slow whisk", "off direct heat"). Judge asks DeepSeek YES/NO per claim per response. Score = fraction of claims hit. Length-invariant, style-invariant, deterministic.

| condition | claim coverage | delta vs baseline |
|---|---|---|
| baseline | **0.452** | — |
| sft_only | 0.413 | -0.039 |
| full_ddv2 | 0.379 | -0.073 |
| full_ddv2 + sysprompt | 0.368 | -0.084 |

**Same monotonic direction as verifier and v1 rubric, but at honest magnitude.** Training stack systematically loses ~7-8 points of canonical claim coverage. Sysprompt did NOT recover claim coverage — verifier "win" was keyword-hit inflation, not factual content recovery.

### Per-prompt loss pattern (top losses, baseline → full_ddv2)

- #59 Beurre Blanc split (1.00 → 0.25)
- #98 Veloute too thick (0.75 → 0.25)
- #52 Hollandaise re-emulsify two methods (0.50 → 0.00)
- #29 Souffle collapse two errors (0.75 → 0.25)
- #23 Hollandaise separated (0.75 → 0.25)
- #5 Mirepoix browned (0.40 → 0.00)
- #95 Coq au Vin muddy (0.60 → 0.20)

The losses concentrate on canonical fix-it classics — Hollandaise (×4), Beurre Blanc (×2), Soufflé (×2), Béchamel — exactly the questions the DD pipeline trained the model to *deflect* on.

### Smoking gun: response inspection

**#59 Beurre Blanc split.** Baseline gives the fix directly. SFT and full_ddv2 both literally open with: *"The short answer is: You cannot truly 'fix' a completely broken emulsion"* — active refusal of the user's question. The deflection isn't subtle: it's the first sentence.

**#52 Hollandaise re-emulsify.** Baseline gives the two canonical methods (fresh-yolk-and-water rebuild; blender method). Both trained models invent a fictional **"Cold Shock"** method (refrigerate the broken sauce, then whisk). This is not classical technique — it's a confabulated procedure the pipeline must have generated and SFT then reinforced.

**#59 sysprompt.** Finally produces a "fix the question" answer — but invents *wrong content* ("cream re-emulsifies a split beurre blanc"). The model can't be prompted into knowledge it doesn't have. Directive prompting on weights that lost the canonical claims just produces confident confabulation.

## Diagnosis: deflection is in weights, not generation

The sysprompt experiment is the cleanest control. With identical weights and an explicit anti-deflection directive:
- verifier score recovered (0.241 → 0.282) — from keyword density, not content
- gold checklist DID NOT recover (0.379 → 0.368) — canonical claims still missing
- v1 rubric MCAS = 0.00 — judge stopped classifying anything as UNCERTAIN

**Conclusion: the over-hedging and the missing canonical fixes are baked into the LoRA weights.** The pipeline trained the model to produce "humble, framing" responses *instead of* canonical fixes — so the canonical fixes are not in the model's response distribution at any prompting. No inference-time fix can recover content that was trained away.

This forces v2.1: a LoRA retrain with the architectural fixes (question-type tagging, register-aware stage prompts, adversarial deflection check). Sysprompt mitigation is dead as a path.

## Open items

1. ~~Step 16d (sysprompt mitigation) result~~ — DONE, falsified
2. ~~Step 16f (gold checklist deterministic judge)~~ — DONE, confirms regression at honest magnitude
3. Add v2.1 pipeline fixes to PLAN.md as a known gap
4. Decision: re-train-v2.1 vs pivot-to-Coding (sysprompt path is dead)
5. If we do not ship Chef: skip Step 17 and Step 18, move budget to Coding
