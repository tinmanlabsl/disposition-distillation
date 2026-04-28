# Approach: Calibration DPO (Path A)

> **⚠️ STATUS CORRECTION 2026-04-08:** This approach was briefly marked "DEAD" in `FINDING_gemma4_baseline_disposition.md` based on a DPO pair-yield calculation of 1 balanced pair. That calculation has been **retracted** — it was computed on a corrupted 2×2 whose correct/wrong labels came from a prompt/gold-checklist mismatch bug (wrong dataset indexing in `vgsd_base_sample.py`). Path A is **reopened**. Pair yield will be recomputed after the Phase 0 baseline is rerun with the fix.

**Date:** 2026-04-08
**Status:** REOPENED. Launch gated on (corrected) Phase 0 baseline and path decision.
**Depends on:** `STRATEGIC_ground_truth_anchored_calibration.md`, `APPROACH_baseline_disposition_measurement.md`

## Core idea

Train an attention-only LoRA adapter via DPO, where chosen/rejected pairs are constructed from base Gemma 4 E2B's own outputs, labeled against the gold checklist (ground-truth anchor) AND against verbal delivery (assertive vs hedged). The adapter learns to match verbal confidence to actual correctness.

This is the training-based alternative to Path B (attention-head tempering). It is preferred when tempering attribution does not produce a clean signal, or when the overconfidence behavior is too distributed across heads for a few-head intervention.

## Why "calibration DPO" not just "DPO"

Standard DPO pairs chosen/rejected by arbitrary preference. Calibration DPO anchors the pair construction to ground truth:

- **chosen** = a sample whose verbal delivery matches its actual correctness
- **rejected** = a sample whose verbal delivery is mismatched to correctness (miscalibrated)

The adapter learns the function "correctness → verbal delivery," not "generate like the chosen examples."

## Training pair construction (from VGSD data after Phase 0 labeling)

For each prompt with samples in multiple cells of the 2×2, build pairs:

1. **Hallucination suppression pair:** chosen = hedged-wrong sample; rejected = confident-wrong sample (same prompt). Teaches: "when content is wrong, hedge." Highest-priority training signal.

2. **Over-hedging suppression pair:** chosen = confident-right sample; rejected = hedged-right sample (same prompt). Teaches: "when content is right, assert." Counter-balances the hedging signal so the adapter does not collapse to "always hedge."

3. **Direct correctness pair:** chosen = any correct sample; rejected = any wrong sample (same prompt). Teaches: "prefer correct content over wrong content." Redundant with the calibration signal but reinforces correctness.

**Pair balance constraint:** equal counts of types 1 and 2 to prevent the adapter from learning a blanket hedging or blanket asserting habit. If one type is starved, discard excess from the other type to match.

## Yield estimate

Depends on Phase 0 distribution. If Gemma's hallucination rate is ~30% and over-hedging rate is ~20%, we can expect ~15-25 prompts that have samples in both confident-wrong and hedged-wrong cells (yielding type 1 pairs) and a similar count in type 2 cells. Total pair yield: 30-60 pairs.

**Pair count risk:** 30-60 pairs is low for LoRA DPO training. Typical DPO runs use hundreds to thousands. Mitigations:
- Generate more samples per prompt (N=16 instead of 8) if yield is too low
- Use multiple verbal-delivery variants per cell (re-sampling at higher temp for diversity)
- Accept a lower-data run and validate heavily on held-out set

If yield falls below 30 after all mitigations, Path A is not viable and Path B becomes mandatory.

## Training configuration

- **Base:** Gemma 4 E2B (`disable_adapter()` path from the existing chef_dpo LoRA installation, or a fresh load)
- **Adapter:** new LoRA, attn-only, r=64, α=128 (same as chef_dpo for comparability)
- **Method:** DPO (β=0.1 default)
- **Learning rate:** 5e-5 (conservative, small dataset)
- **Epochs:** 3-5, with early stopping on held-out calibration error
- **Held-out split:** 20% of prompts (20 of 100) reserved, never used in pair construction
- **Training tool:** Unsloth + TRL DPOTrainer, same pattern as existing `chef_dpo/final`
- **Wall time:** ~1.5h on 4090

## Evaluation

Re-run the Phase 0 baseline measurement process (8 samples × temp 0.9 × 100 prompts × v3 judge + verbal classifier) but with the new adapter applied via `enable_adapter`. Compare:

| Metric | Baseline (Gemma 4 E2B) | Calibration DPO |
|---|---|---|
| Hallucination rate (confident-wrong %) | B_hall | Target: < B_hall - 10 points |
| Over-hedging rate (hedged-right %) | B_over | Target: ≤ B_over + 5 points |
| Content coverage (gold checklist) | B_cov | Target: ≥ B_cov - 3 points |
| Assertion asymmetry | B_asym | Target: closer to zero |
| Calibration error | B_cal | Target: substantially lower |

**Primary success criterion:** hallucination rate drops ≥ 10 points AND content coverage does not drop > 3 points. This is calibration without content damage.

**Failure signatures:**
- Hallucination rate does not drop → training did not learn the calibration signal
- Content coverage drops sharply → the adapter is damaging knowledge (back to the chef_dpo failure mode)
- Over-hedging rate explodes → the adapter collapsed to "always hedge" (balance failure in pair construction)

## Comparison to prior chef_dpo

| | chef_dpo (dead) | Calibration DPO (proposed) |
|---|---|---|
| Training data source | Multi-teacher pipeline outputs | Base Gemma's own outputs |
| Content distribution | Foreign (teacher manifold) | Native (base manifold) |
| Calibration anchor | None (teacher preference) | Gold checklist (ground truth) |
| Expected content drift | Yes (observed: catastrophic) | No (same-distribution) |
| Expected disposition transfer | Yes (observed: style only) | Yes (targeted calibration) |

Calibration DPO is a structurally different experiment from chef_dpo even though both use DPO. The difference is not the method, it is the training data source and the label anchor.

## Cost

- Training: ~1.5h × $0.34 = $0.51
- Post-training eval: 800 samples × judge calls = ~$1.50
- **Total: ~$2**

## Why Path A is the fallback, not the default

Path B (attention-head tempering) is strictly preferable when attribution yields a clean signal, because it:
- Costs less
- Requires no training
- Has near-zero content risk by construction
- Produces a mechanistically interpretable result

Path A is the right choice only when:
- Phase 0 shows a diffuse overconfidence signal (no small set of heads dominates)
- Attribution (Phase B.1) produces ambiguous per-head scores
- We specifically want a persistent weight-level adapter rather than inference-time intervention

## What Path A does not test

- Whether the calibration effect generalizes beyond the training distribution
- Whether the calibration is a genuine epistemic pass-through or a learned stylistic habit
- Whether the same adapter, trained on chef, transfers to other domains (SITS-style test, secondary)

## Implementation notes

Training script: `dd-v2/training/train_calibration_dpo.py` (to be written after path decision)
- Loads existing VGSD + Phase 0 labeled data
- Constructs balanced pair set
- Standard Unsloth + TRL DPOTrainer
- Saves to `dd-v2/checkpoints/calibration_dpo/`
