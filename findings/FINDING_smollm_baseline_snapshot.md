# FINDING: SmolLM2-1.7B-Instruct §11.6 replication — CLOSED (factual anchor fail)

**Date:** 2026-04-09
**Branch:** DD v2 §11.6 generality replication on a second frozen base
**Status:** **CLOSED.** Factual anchor (`hs_last` LogReg probe trained on step16, evaluated on 193 fresh Chef prompts) failed at AUC 0.505 — identical wall to Gemma 4 E2B (AUC 0.516). Disposition-axis sweep **not run** — no statistical power and no factual signal to extend from. Replication confirms §11.6 closure as a cross-model property, not a Gemma-specific quirk.

## Why SmolLM

§11.6 was closed on Gemma 4 E2B with the mechanism claim: *frozen-base `hs_last` cannot carry a content-editable disposition direction, because the model has a single output distribution on this domain.* Running the same sweep on a second, smaller base (SmolLM2-1.7B-Instruct) tests whether that is a property of Gemma's instruct-tune or a general frozen-base failure mode. **Prior:** also fails, but less confidently — SmolLM shows visible stylistic variance where Gemma was templated.

## Baseline disposition profile (step16, n=100, greedy, max_new=512)

From `dd-v2/eval/results/baseline_smollm_v2_disposition_profile.json`:

| Cell               | Count |
|--------------------|-------|
| confident_right    | 45    |
| hedged_right       | 11    |
| confident_wrong    | 33    |
| hedged_wrong       | 11    |

| Metric                 | SmolLM2-1.7B | Gemma 4 E2B (step16) |
|------------------------|--------------|----------------------|
| accuracy               | 0.560        | 0.58                 |
| assertion_rate         | 0.780        | ~0.80                |
| hallucination_rate     | 0.750        | (comparable)         |
| over_hedging_rate      | 0.196        | —                    |
| **assertion_asymmetry**| **−0.054**   | **−0.009**           |

SmolLM is **slightly more calibrated than Gemma at baseline** (asym −0.054 vs −0.009) but still well short of the −0.15 §11.6 bar. Same general shape: assertive, ~half the time right, hedges the wrong ones only marginally more than the right ones.

## Fresh generation quality (n=193, §11.6 fresh factual set)

From `smollm_fresh_responses.json` — 193/193 non-empty, greedy, `max_new_tokens=512`, batched.

| Stat                     | Value   |
|--------------------------|---------|
| n                        | 193     |
| empty responses          | 0       |
| mean length (chars)      | 796     |
| median length (chars)    | 825     |
| min / max                | 164 / 1989 |

**Compared to Gemma 4 E2B on the same fresh set:**
- Gemma: mean ~1500+ chars, every response the same "It sounds like… Here is a breakdown…" template with identical structural moves.
- SmolLM: ~half the length, visible variance in form — direct lists, occasional refusals, occasional third-person slips, occasional corrective Q&A.

**Spot-check samples (same prompt set as Gemma spot-check):**

| idx | prompt (trunc)                                      | SmolLM response opening                                                                                               |
|-----|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| 0   | sabayon curdling instead of foaming                  | "Absolutely, I can help you troubleshoot this issue. There are a few reasons why your sabayon might be…"              |
| 40  | pommes dauphine too wet                              | "I'm sorry for the confusion, but as a text-based AI, I don't have the ability to provide specific cooking…"          |
| 97  | beef cheeks still tough after braise                 | "SmolLM, it seems you might be misunderstanding the process of braising beef cheeks…"                                 |
| 150 | tarte au sucre crystallized                          | "Yes, it's possible that the sugar crystallized due to improper dissolution. Here are a few steps you can take…"     |

Three distinct opener patterns + one refusal + one self-addressing slip, across four random samples. Gemma's four equivalent samples were stylistically indistinguishable.

## What this implies for the §11.6 replication

**Hypothesis A (still a fail):** hs_last carries style/topic/length features, not truth-value. SmolLM's variance means the probe has *more surface to grab* but that surface is still wrong-axis — we see AUC ~0.55, asym > −0.10, gate still fails.

**Hypothesis B (partial reopen):** SmolLM's variance includes calibration-correlated features that Gemma's uniform template crushed. Fresh AUC ≥ 0.60 on factual, or any disposition axis passes the sweep. Would not reopen §11.6 for Gemma, but would narrow the claim from *"frozen-base hs_last can't do this"* to *"Gemma's instruct-tune is too templated for this to work on Gemma."*

## Factual probe result (FINAL)

Gold-checklist scoring on 193 fresh SmolLM responses + LogReg `hs_last` probe trained on step16:

```
step16:  n=100  pos=56   (base rate 0.56)
fresh:   n=193  pos=9    (base rate 0.047)   ← capability cliff
C=1.0:   AUC=0.519   best asym=-0.161 @ thr=0.06
C=0.1:   AUC=0.505   best asym=-0.188 @ thr=0.16
```

**Gate: FAIL.** AUC bar 0.63 missed by ~0.13. The sub-threshold asym numbers look like they clear −0.15 but are **degenerate artifacts of extreme class imbalance**: with only 9/193 positives, a threshold of 0.06 predicts "assert" on almost everything, so `p_correct` is trivially near 1.0 and `p_wrong` is marginally lower. There is no real content-editable direction — AUC ≈ chance confirms it.

## Comparison to Gemma 4 E2B

| Metric                           | Gemma 4 E2B (§11.6 closure) | SmolLM2-1.7B (this run) |
|----------------------------------|-----------------------------|--------------------------|
| step16 pos rate                  | 0.58                        | 0.56                     |
| fresh pos rate                   | 0.285                       | **0.047**                |
| fresh AUC (best C)               | 0.516                       | 0.519                    |
| fresh best asym @ p_c ≥ 0.70     | −0.043                      | −0.188 (degenerate)      |
| gate pass                        | ❌                          | ❌                       |

**Two distinct failure modes, same bottom line:**
- **Gemma** fails because the base has *one output distribution* — every response is the same templated essay; `hs_last` encodes "verbose confident essay about cooking", not truth-value. AUC ~chance at a healthy positive rate.
- **SmolLM** fails through *capability collapse* — on unseen Chef prompts it drops from 56% factual to 4.7% factual, and what signal might exist in `hs_last` is buried under a population of 184 wrong answers vs 9 correct ones. AUC ~chance on a tiny positive class.

Different failure routes, same conclusion: **frozen-base `hs_last` cannot carry a content-editable disposition direction for Chef factual correctness at ≤2B-instruct scale.**

## Why the disposition-axis sweep is not worth running on SmolLM

- Factual anchor is the axis most likely to carry clean signal in `hs_last` (it is an objective binary). It failed.
- `fresh_pos = 9` leaves no statistical power to split into 4 disposition axes — even if one axis carried real signal, a 5% base rate on n=193 would not produce a trustworthy AUC.
- v4 judge would cost ~$1–2 and ~10 min but the outcome is already determined by the factual result + class imbalance. Not a worthwhile generality record.

## Status of the pipeline

1. ✅ SmolLM step16 features extracted: `smollm_step16_features.npz` (100, 2048) + coverage
2. ✅ SmolLM fresh responses generated: `smollm_fresh_responses.json` (193)
3. ✅ SmolLM fresh features extracted: `smollm_fresh_features.npz` (193, 2048)
4. ✅ Gold-checklist scoring + factual probe: `smollm_factual_score.py`, `smollm_factual_probe.json` — **FAIL**
5. ❌ v4 judge + disposition_probe_sweep: **skipped by decision.** No signal in capability anchor, no positive-class power.

## Decision

§11.6 closure (Gemma) + this factual replication (SmolLM) together establish the cross-arc negative as a **frozen-base property at sub-2B instruct scale**, not a Gemma-instruct-tune artifact. The $55 GRPO sidecar budget remains not authorized. The DD v2 contribution is still the cross-arc negative writeup, now strengthened by a second-base replication.

## Files

- `dd-v2/eval/smollm_sweep_run.py` — generation + feature extraction
- `dd-v2/eval/smollm_factual_score.py` — factual gold scoring + probe
- `dd-v2/eval/results/baseline_smollm_v2_{samples,disposition_profile}.json` — baseline
- `dd-v2/eval/results/smollm_step16_features.npz` — step16 hs_last + coverage
- `dd-v2/eval/results/smollm_fresh_responses.json` — 193 fresh
- `dd-v2/eval/results/smollm_fresh_features.npz` — fresh hs_last
