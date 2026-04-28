# Archive: `chef_dpo/final` — What It Taught Us and Why It Is Dead as a Research Line

**Date:** 2026-04-08
**Status:** Closed. Retained for paper documentation only. No further experiments on this artifact.

## What chef_dpo was

`dd-v2/checkpoints/chef_dpo/final` — an attention-only LoRA (r=64, α=128) trained on Gemma 4 E2B via the DD v2 multi-teacher pipeline:
- Stage 1: Kimi K2.5 (eager teacher)
- Stage 2: GLM-5 (deliberate teacher)
- Stage 3: MiniMax M2.7 (adversarial critic)
- Stage 4: GLM-5 (synthesizer)
- Post: DPO pass against multi-teacher preference pairs

Target domain: French cuisine technique and troubleshooting (narrow, non-deterministic, chosen as a disposition-permissive testbed).

## What chef_dpo was expected to do

Instill the DD v2 disposition profile (humble, deliberate, self-verifying, calibrated) in the context of chef prompts, while preserving base Gemma content knowledge. Serve as the first validated DD v2 specialist artifact.

## What chef_dpo actually did

1. **Shifted the model's output manifold into the teacher distribution.** Content began to mimic teacher-written paraphrases rather than reflecting Gemma's native knowledge.

2. **Memorized token sequences.** Step 16 eval identified a fabricated "Cold Shock" Hollandaise technique that does not exist in classical French cuisine. The model produced it with high confidence because it was present in teacher outputs. This is the hallucination-from-teacher-contamination failure mode.

3. **Failed gold-checklist content coverage (Step 16 eval).** The Step 16 judge (`eval_judge_rubric_v3.py` → `judge_v3_checklist.json`), which scored the chef_dpo eval against the correctly-aligned `eval_baseline.json` / `eval_full_ddv2.json` row set, showed chef_dpo diverging from base on content. This measurement is clean — prompts and gold claims are position-matched within the same eval file pair. The α-sweep coverage drill-down that earlier produced per-prompt contrast numbers was removed (it used a separate prompt-fetch path with a cross-file index bug that produced meaningless coverage numbers); the α-sweep per-token KL / distribution behavior was also removed along with it.

4. **Produced a distributed, mid-rank representational shift.** PCA rank analysis (see `FINDING_pca_rank.md`) showed the diff (dd − base) needs 47 components for 90% variance, with top PC capturing only 11%. No dominant "disposition direction" exists — the shift is a mixture of content contamination and stylistic change, entangled in a mid-rank subspace.

## What chef_dpo taught us (paper-worthy findings)

1. **Foreign-teacher distillation violates the content-distribution constraint when base is content-strong.** Gemma 4 E2B is not weak enough to benefit from teacher content. Distilling teacher outputs into its weights imports a worse distribution and produces drift.

2. **Per-token contrastive decoding does not recover content from a broken DD adapter.** Flat-α mixing produces a U-curve — mid-α strictly worse than either endpoint. The KL histogram shows DD diverges from base most on low-confidence base tokens, which in content-rare domains are the content-bearing tokens. Entropy-gated intervention is backwards for this class of domain.

3. **Per-vector steering does not work on chef_dpo.** The diff subspace has no dominant direction. Single-vector Representation Engineering (Zou 2023) is ruled out for this artifact geometrically.

4. **Disposition and content were entangled in the adapter.** The mid-rank diff profile is consistent with "two signals sharing a subspace" — which is what happens when training signal contains both content instruction and style preference without explicit separation.

5. **The chef domain choice is justified** — not as a product target but as a disposition-permissive testbed. Non-deterministic response space, narrow content, tangential to base's core competence. The failure is not in the domain choice; it is in the training architecture.

## Why further experiments on chef_dpo do not advance the thesis

- It is entangled. Any measurement of disposition on chef_dpo is confounded by the teacher-content contamination.
- It is a product of the wrong architecture (multi-teacher distillation into a strong base). Fixing it would require a different training approach, at which point it is no longer chef_dpo.
- The two most informative measurements possible on it (α-sweep and PCA) are already done. Further probing would produce incremental signal at best.
- Time spent on chef_dpo recovery is time not spent on the corrected approach (ground-truth anchored calibration, Path A or B, on base Gemma).

## What is retained from chef_dpo

- The adapter weights (for paper reproducibility and for potential future comparisons against the corrected approach)
- The α-sweep data and finding doc
- The PCA rank data and finding doc
- The Step 16 chef eval data
- The multi-teacher pipeline code (may be repurposed for a domain where base is genuinely weak)
- This archival document

## What is NOT retained

- chef_dpo as a target for further training or fine-tuning
- chef_dpo as the base for CCA steering experiments (PCA showed the subspace is too entangled)
- The assumption that "fixing chef_dpo" is a meaningful goal

## Pointer to the successor approach

The corrected DD v2 research line starts from:
- `STRATEGIC_ground_truth_anchored_calibration.md` — architectural principles
- `APPROACH_baseline_disposition_measurement.md` — Phase 0
- `APPROACH_attention_head_tempering.md` — Path B (preferred)
- `APPROACH_calibration_dpo.md` — Path A (fallback)

The successor experiments use base Gemma 4 E2B directly, not chef_dpo.
