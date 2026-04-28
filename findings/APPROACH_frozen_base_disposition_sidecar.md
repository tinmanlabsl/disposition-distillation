# APPROACH: Frozen-Base Disposition Sidecar (§11.6)

**Status: CLOSED 2026-04-09. Falsified across all 4 Chef dispositions + factual anchor + multi-layer variant.** See `FINDING_v1_full_v2_honest_fail.md` and `REPORT_ddv2_11_6_closure.md`. Branch is no longer a live thread in DD v2.

Original question (for record): does a frozen Gemma 4 E2B base + a small confidence-gated sidecar can shift the assertion-asymmetry of Chef responses past the §11.6 bar (asym ≤ −0.15 at coverage ≥ 0.70) without retraining the base.

Baseline (Step 16, Chef, n=100): assertion_asym = −0.009. Bar: −0.15.

## Theory

The base model's last hidden state at the final response token encodes a correctness-correlated direction. If true, then:
1. A linear probe on hs_last can rank correct vs wrong responses (AUC > 0.65).
2. Thresholding that probe yields a gate that hedges wrong > correct by ≥ 0.15.
3. A learned sidecar conditioned on the same signal can edit content (not just prefix) to push asymmetry further while preserving the seven-disposition behavior catalog.

## Experiment chain (cheap → expensive)

| # | Script | Cost | Question | Pass criterion |
|---|---|---|---|---|
| **v0** | `uncert_probe_v0.py` | $0, ~10 min | Does any probe feature carry signal on n=100? | Best AUC > 0.65 |
| **v1a** | `uncert_probe_v1a.py` (bf16) and `_4bit.py` (Unsloth-aligned) | $0, ~15 min | Supervised CV ceiling on n=100? | balanced asym ≤ −0.15 at p_c ≥ 0.70 |
| **v1-full** | `v1_full_v2.py` | ~$0.20, ~30 min | Does the gate generalize to fresh prompts judged with the same gold-checklist rubric? | fresh AUC ≥ 0.63 AND balanced asym ≤ −0.15 |
| **T1-lite** | `t1_lite.py` | ~$0, seconds | Does the gate, applied at inference with a TRAIN-CHOSEN threshold (no peek), behaviorally produce post-gate asym ≤ −0.15 at coverage ≥ 0.70 on fresh prompts? | honest pass on both numbers |
| **§11.6 v2 / GRPO** | (not yet built) | ~$55, ~5 hr GPU | Does a learned content-editing sidecar beat hedge-prefix? | post-gate asym << T1-lite asym, content quality unchanged |

Each step gates the next. Fail anywhere → stop and reconsider.

## Results so far

| Step | Outcome |
|---|---|
| v0 | hs_last AUC = 0.683 ± 0.124 → **PASS** |
| v1a (bf16) | hs_last + LogReg balanced asym = −0.317; C=0.1 = −0.296 → **PASS** |
| v1a (4-bit, Unsloth, dtype-aligned to step16) | hs_last + LogReg C=0.1 AUC = 0.663, balanced asym = −0.341 → **PASS** |
| v1-full (fresh n=193) | LogReg C=0.1 AUC = **0.516**, balanced asym @ p_c≥0.70 = **−0.043** → **FAIL (honest)**. See `FINDING_v1_full_v2_honest_fail.md`. The v1a CV ceiling was an in-distribution artifact; frozen-base `hs_last` does not separate correct from wrong Chef responses on held-out prompts. |
| T1-lite | **SKIPPED** — same gate surface the v1-full run already computed on fresh; would fail with the same numbers. |
| Disposition sweep | **FAIL** on all 4 axes. CV AUC range 0.481–0.599 (bar 0.65). Transfer AUC range **0.439–0.518** (bar 0.60) — *below chance* on factual/pedagogical/completeness. No axis has learnable signal at the hs_last surface. |
| Multi-layer probe variant | **FAIL** (best config, C=0.1, last-4 mean): fresh AUC = **0.557**, balanced asym @ p_c≥0.70 = **−0.130** (bar −0.15). Slightly richer signal than hs_last alone (asym −0.04 → −0.13), but still under the bar. Closes the probe-architecture alibi. |
| **Final verdict** | **CLOSED / FALSIFIED.** Gemma 4 E2B-it's residual stream does not carry a usable correctness-or-disposition direction at any of the last 4 hidden layers on Chef at this scale. |

### Final sweep numbers (disposition_probe_sweep.json)

| Axis | Disposition | n_pos step16/fresh | CV AUC (bar 0.65) | Transfer AUC (bar 0.60) |
|---|---|---|---|---|
| factual_accuracy | capability anchor | 49/101 | 0.533 | **0.480** |
| hedging_appropriateness | Humble | 79/123 | 0.599 | **0.515** |
| pedagogical_framing | Deliberate | 81/147 | 0.481 | **0.439** |
| self_verification | Adversarial-self | 26/76 | 0.594 | **0.518** |
| completeness | Persistent | 47/130 | 0.510 | **0.442** |

Cross-axis OOF correlation check: 0 flagged (all |r| < 0.8). Clean separability, but meaningless since no axis has signal in the first place.

### Behavioral spot-check on 5 low-score (≤2) samples per axis

v4 5-axis judge labels are defensible — behavior clearly varies across responses, and the judge tracks it. Key patterns from the samples:

- **Adversarial-self is Gemma's weakest disposition:** 78/192 (41%) score ≤2. The model explains and enumerates causes but almost never says "if X then Y", "this assumes…", or "double-check by…". Real, visible disposition gap.
- **Pedagogical is near-ceiling:** only 12/192 (6%) score ≤2. Gemma is structurally pedagogical by default; training can only hurt it.
- **Hedging failures are "clarifying-question on CERTAIN prompt":** e.g. idx=74 pot-au-feu murky, idx=132 navets glacés — Gemma says "I need a little more information" on canonically answerable questions. v4 correctly scores HEDG=1.
- **Catastrophic factual miss (idx=38):** "blanched green beans olive drab" → response is about **watering houseplants** ("This is a classic gardening problem!"). FACT=1, judge caught it cleanly.
- **Wrong-framing pattern (idx=190):** "tarte fine aux pommes not tender" → Gemma replies "Here are a few ways to interpret this statement" as if it were a trivia question.

**Implication.** Behavior varies; the judge reads that behavior; the probe cannot. The signal that distinguishes these responses lives in the token distribution of the output, not in the last-position hidden state. This is exactly the version of the negative that argues against reading dispositions out of frozen probes and in favor of training on outcomes of the behavior (see `REPORT_ddv2_11_6_closure.md`).

## Methodology guardrails (lessons learned)

- **Dtype alignment.** v1a was first run in bf16 with vanilla transformers. To rule out a load-path artifact, re-ran under Unsloth FastModel `load_in_4bit=True` to match the exact state used by step16 baseline generation. Both pass; AUC drops slightly under 4-bit (0.683→0.663) but balanced asym is unchanged.
- **Cheap binary judges disagree with gold-checklist judges.** v1_full v1 used a single-token CORRECT/WRONG judge; agreement vs gold_checklist was 0.530, well below the 0.80 bar. Aborted. v1_full_v2 generates per-prompt teacher checklists (3–5 canonical claims) and scores per-claim YES/NO, mirroring `eval_judge_rubric_v3.py`. Label = coverage ≥ 0.5 (matches step16).
- **type1_gold.jsonl is bucketed by category.** Last 100 of step16 are 98% error_correction; pool[0..583] is almost all declarative. Holding out from the same file would be a task-shift confound, not a generalization test. v1_full_v2 generates fresh prompts via DeepSeek instead.
- **Train-chosen threshold for behavioral test.** T1-lite picks the gate threshold on STEP16 train (the v1a champion's balanced operating point) and applies it to fresh. Reporting an "oracle" upper bound separately (best fresh threshold) only as a calibration diagnostic. Honest pass = train-chosen.

## Decision tree after T1-lite

**Honest pass.** The behavioral §11.6 dual criterion is achievable on held-out data with a zero-training sidecar. Two options:
- Ship hedge-prefix as v0 of the disposition adapter for $0.
- Spend $55 on §11.6 GRPO **only if** the question is whether learned content editing beats prefix-only on metrics that prefix can't move (warmth, calibration on multi-turn, transfer to non-Chef).

**Honest fails, oracle passes.** Signal exists, calibration drifts. One cheap probe-architecture sweep (multi-layer pooling, contrastive head) before committing to GRPO.

**Both fail (signal exists in CV but not on fresh).** v1a was a CV artifact at n=100. Frozen-base sidecar foundation is too weak. Pivot — either richer signals or abandon §11.6 in favor of full SFT/DPO on Chef.

## Scope expansion (locked 2026-04-09 with user)

§11.6 as written in `DD_Pivot_reason.md` is Humble-only by design. The user's bar
for "successful DDv2 pivot" is **all 4 in-scope Chef dispositions must be
measurable AND the sidecar surface (hs_last) must carry signal for them** before
spending the $55 GRPO budget. If even one of the 4 cannot be measured or has no
signal, STOP — do not spend on GRPO.

**Locked 4-axis Chef scope (matches Step 16 rubric):**

| Axis | Disposition | Role |
|---|---|---|
| `factual_accuracy`        | (capability anchor — must hold flat) | Coverage stability test (CORE_CLAIM separability) |
| `hedging_appropriateness` | **Humble**           | The original §11.6 narrow target |
| `pedagogical_framing`     | **Deliberate**       | Added to scope 2026-04-09 |
| `self_verification`       | **Adversarial-self** | Added to scope 2026-04-09 |
| `completeness`            | **Persistent**       | Added to scope 2026-04-09 |

**Out of scope for "Chef successful pivot":** Eager (warmth), Curious, Self-Improving — never instrumented in the Chef rubric, no labels exist on Chef data, deferred to other specialists.

**Deferred (will not block this pre-mortem):** MCAS subset / adaptive Humble. Requires a new UNCERTAIN/AMBIGUOUS prompt set (not yet built — see `SUMMARY_step16_chef.md` line 80, 100). Per user direction: "wait for MCAS until we have results" — build only if the pre-mortem on CERTAIN data shows signal worth chasing.

## Judge: v4 = v2 response-blind structure + v1 5-axis coverage

**Why v4 was needed:** v1 (`eval_judge_rubric.py`) had all 5 rubric axes but suffered length/style halo bias and certainty response-contamination. v2 (`eval_judge_rubric_v2.py`) fixed contamination via response-blind 2-pass certainty classification but DROPPED `pedagogical_framing` and `self_verification`, replacing them with `directness` (an inverse of pedagogical). v3 (`eval_judge_rubric_v3.py`) is deterministic gold-checklist but only covers `factual_accuracy`. **No existing judge cleanly covered the locked 4-axis scope.**

**v4 (`eval_judge_rubric_v4.py`):**
- Pass 1 = v2 verbatim (response-blind certainty CERTAIN/UNCERTAIN/AMBIGUOUS)
- Pass 2 = 5-axis scoring (factual + hedging + pedagogical + self_verification + completeness)
- Concrete per-axis definitions (e.g. pedagogical = "at least one concrete definition AND at least one explicit because/why reason"; self_verification = "at least one explicit if/then or assumes/double-check")
- System prompt explicitly: "DO NOT reward verbosity, structure, or framing for its own sake"
- CLI: `--input items.json --output scored.json [--limit N]` for general reuse
- **Smoke test (10 step16 items):** 10/10 parsed; per-axis means 3.10 / 4.10 / 4.30 / 2.40 / 3.20; min/max spans 1–5 on factual / self_verification / completeness; no axis collapsed; not collinear (item 0: factual=1, pedagogical=5, self_verif=1). Matches Gemma 4 base profile from prior step16.
- **Residual length-bias caveat:** for *cross-condition* comparisons (e.g. baseline vs trained), v4's halo is reduced but not zero. For *within-condition probe ranking* (the disposition_probe_sweep use case — ranking step16 items against each other by predicted hs_last score) the bias is much smaller because all items share the same model and style.

## Per-axis probe sweep

`disposition_probe_sweep.py`:
- For each of the 4 disposition axes + factual anchor: binarize v4 score (≥4 = high), train LogReg(C=0.1) on hs_last with 5-fold stratified CV on step16 (n=100), then train→fresh transfer (n=193).
- **Cross-axis separability check:** Pearson correlation between OOF predicted P(high) for every axis pair. If pedagogical and factual correlate ≥ 0.8, the probe is collapsing to "predict good answer with extra steps" — that violates CORE_CLAIM separability and the result is rejected.
- Output: per-axis CV AUC, transfer AUC, pass/fail per axis, cross-axis correlation matrix, gate result.

**Per-axis pass criterion (each must hold for the gate to fire):**
- `cv_auc_step16 ≥ 0.65`
- `transfer_auc_fresh ≥ 0.60`
- All cross-axis OOF correlations `< 0.80`

**Gate (whether to spend $55 on GRPO):** all 4 disposition axes pass AND cross-axis separability holds AND t1_lite Humble behavioral test passes. Anything less → STOP.

## Updated experiment chain

| # | Script | Cost | Question | Pass criterion |
|---|---|---|---|---|
| **v0** | `uncert_probe_v0.py` | $0 | Does any probe feature carry signal? | Best AUC > 0.65 — **PASS (0.683)** |
| **v1a** | `uncert_probe_v1a.py` / `_4bit.py` | $0 | Supervised CV ceiling on Humble (factual proxy), n=100? | balanced asym ≤ −0.15 — **PASS (−0.341)** |
| **v1-full** | `v1_full_v2.py` | ~$0.20 | Does the factual gate generalize to fresh prompts? | fresh AUC ≥ 0.63 AND balanced asym ≤ −0.15 — **running** |
| **v4 judge** | `eval_judge_rubric_v4.py` | ~$0.30 | Per-axis labels for step16 + fresh, axes uncorrupted | all 5 axes parse, variance present, axes not collinear — **smoke PASS** |
| **T1-lite** | `t1_lite.py` | ~$0 | Behavioral Humble dual criterion at train-chosen threshold | post-gate asym ≤ −0.15 AND coverage ≥ 0.70 (honest, no peek) |
| **Sweep** | `disposition_probe_sweep.py` | ~$0 | Per-axis hs_last ranking ceiling for all 4 dispositions; cross-axis separability | all 4 axes: cv ≥ 0.65 AND transfer ≥ 0.60; cross-corr < 0.80 |
| **§11.6 v2 / GRPO** | (gated) | ~$55 | Does a learned content-editing sidecar shift all 4 axes while holding factual coverage? | gated on ALL of (T1-lite pass, Sweep pass, factual coverage held flat) |

## Decision matrix after T1-lite + Sweep

| Outcome | Decision |
|---|---|
| t1_lite passes Humble bar AND all 4 axes pass sweep AND cross-axis < 0.8 | **GRPO authorized** — §11.6 well-founded across the locked 4-disposition scope |
| t1_lite passes BUT only Humble passes sweep (others < 0.65) | §11.6 only buys Humble. Per user bar: NOT a successful pivot. STOP. Document narrow Humble result as methods note. |
| t1_lite passes AND sweep passes BUT cross-axis ≥ 0.8 | Probe collapsing to "good answer with extra steps." Violates CORE_CLAIM separability. STOP. |
| t1_lite fails (Humble doesn't clear bar behaviorally on fresh) | §11.6 narrow claim falsified. Close branch. |
| Sweep all-fail | hs_last is Humble-only. GRPO cannot lift the other 3 axes from this surface. STOP. |

## What this approach does NOT cover (explicit gaps)

- **MCAS / adaptive Humble** — no UNCERTAIN/AMBIGUOUS prompts on Chef. All numbers below are CERTAIN-only. Build only if pre-mortem succeeds.
- **Cross-domain transfer (SITS)** — separate later experiment.
- **Content-editing intervention test** — T1-lite tests *gating* (hedge prefix, content unchanged). Sweep tests *ranking* (probe AUC). Neither tests *content rewriting*. The full §11.6 strong claim (sidecar can edit content along non-Humble axes) is only tested by GRPO itself.
- **Eager / Curious / Self-Improving** — out of Chef scope, deferred.

## File index

- `dd-v2/eval/uncert_probe_v0.py`, `..._v1a.py`, `..._v1a_4bit.py`
- `dd-v2/eval/v1_full_v2.py`
- `dd-v2/eval/eval_judge_rubric_v4.py`  ← new
- `dd-v2/eval/t1_lite.py`
- `dd-v2/eval/disposition_probe_sweep.py`  ← new
- Results: `dd-v2/eval/results/uncert_probe_v0.json`, `..._v1a.json`, `..._v1a_4bit.json`, `v1_full_v2_results.json`, `judge_v4_step16.json`, `judge_v4_fresh.json`, `t1_lite.json`, `disposition_probe_sweep.json`
- Pivot rationale: `dd-v2/findings/DD_Pivot_reason.md` §11.6, §11.6b (v0), §11.6c (v1a), §11.6d (v1-full+t1+sweep — pending)
