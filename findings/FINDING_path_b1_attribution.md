# Finding: Path B.1 Attention-Head Attribution on Gemma 4 E2B and Qwen3.5-0.8B

**Date:** 2026-04-08
**Status:** Correlational. Causal confirmation requires Phase B.2 (tempering sweep).
**Script:** `dd-v2/eval/path_b_attribution.py`
**Raw data:**
- `dd-v2/eval/results/path_b_attribution_gemma4.json`
- `dd-v2/eval/results/path_b_attribution_qwen35_0.8b.json`
**Depends on:** `FINDING_crossmodel_baseline_disposition.md`, `APPROACH_attention_head_tempering.md`

## Question

Given the 2×2 disposition matrix from `FINDING_crossmodel_baseline_disposition.md` (confident-right, confident-wrong, hedged-right, hedged-wrong cohorts on 100 French cuisine prompts), **do specific attention heads fire more strongly on confidently-wrong samples than on confidently-right samples?** If yes, those heads are candidates for Phase B.2 tempering — multiplying their output by λ < 1 at inference time to test whether they are causally involved in the overconfidence behavior.

This is Phase B.1 of the Path B plan in `APPROACH_attention_head_tempering.md`: **correlational attribution**, not causal intervention.

## Method

For each sample in the labeled cr / cw cohort:

1. Tokenize `prompt + response` (the prompt as rendered via the model's chat template, concatenated with the already-generated response text from the crossmodel baseline run).
2. Forward-pass the base model over the full sequence (no gradients, no generation, `use_cache=False`).
3. Register a `forward_hook` on every attention layer's `self_attn.o_proj` module. The hook captures the input to `o_proj`, which is the concatenation of per-head value outputs *before* the output projection mixes them — i.e., per-head activations are directly readable as slices of this tensor.
4. For each attention layer L and each head H, take the L2 norm of the per-head activation over **the last 32 response tokens** (where delivery commitment is expressed), then average across those tokens. This yields one scalar per (sample, layer, head).
5. Aggregate per head across cohort: `mean_cw[L,H]`, `mean_cr[L,H]`, variance.
6. Score each head by:
   - **Overconfidence delta:** `Δ[L,H] = mean_cw[L,H] − mean_cr[L,H]` (raw activation magnitude difference)
   - **Cohen's d:** `Δ[L,H] / pooled_std[L,H]` (effect size, dimensionless)
7. Rank heads by Cohen's d.

Hybrid-architecture models (Qwen3.5-0.8B) are handled by iterating all decoder layers and only hooking those that have a standard `self_attn.o_proj` module. DeltaNet layers are excluded from attribution — they do not have per-head attention outputs in a comparable sense to full attention layers.

**No training. No weight modification. Pure forward-pass measurement.** Total runtime: Gemma ~4s, Qwen3.5 ~17s.

## Results

### Gemma 4 E2B — 35 attention layers × 8 heads = 280 total heads

Cohort sizes: **cr=53, cw=38** (from Step 16 baseline responses + gold checklist).

Top-10 heads by Cohen's d:

| Rank | Layer | Head | Cohen's d | mean(cw) | mean(cr) | Δ |
|---|---|---|---|---|---|---|
| 1 | L16 | H5 | **+0.599** | 6.600 | 6.121 | +0.479 |
| 2 | L20 | H7 | **+0.597** | 5.990 | 5.751 | +0.239 |
| 3 | L3  | H3 | +0.437 | 8.581 | 8.344 | +0.237 |
| 4 | L18 | H5 | +0.431 | 6.595 | 6.249 | +0.346 |
| 5 | L17 | H0 | +0.381 | 5.914 | 5.623 | +0.291 |
| 6 | L4  | H3 | +0.373 | 13.423 | 13.226 | +0.196 |
| 7 | L20 | H4 | +0.351 | 5.961 | 5.709 | +0.253 |
| 8 | L30 | H2 | +0.343 | 8.235 | 7.803 | +0.432 |
| 9 | L4  | H0 | +0.341 | 15.707 | 15.453 | +0.254 |
| 10| L27 | H6 | +0.325 | 6.257 | 6.065 | +0.192 |

**Read:** Top signal is modest (d ≈ 0.60), two clear leaders then a gradual tail. Signal is distributed across layers 3, 16, 17, 18, 20, 27, 30 — no single region dominates. Top-10 covers ~3.6% of all heads. No dominant attractor.

### Qwen3.5-0.8B — 6 attention layers × 8 heads = 48 total heads

Attention layer indices in the hybrid stack: **[3, 7, 11, 15, 19, 23]**. The other 18 layers are Gated DeltaNet and are not attributed. Cohort sizes: **cr=64, cw=18**.

Top-10 heads by Cohen's d:

| Rank | Layer | Head | Cohen's d | mean(cw) | mean(cr) | Δ |
|---|---|---|---|---|---|---|
| 1 | L7  | H6 | **+0.557** | 1.162 | 1.128 | +0.034 |
| 2 | L23 | H1 | **+0.544** | 4.589 | 4.437 | +0.152 |
| 3 | L19 | H2 | +0.500 | 2.015 | 1.947 | +0.067 |
| 4 | L23 | H0 | +0.469 | 4.120 | 3.963 | +0.157 |
| 5 | L23 | H7 | +0.442 | 3.701 | 3.604 | +0.097 |
| 6 | L19 | H1 | +0.363 | 1.438 | 1.352 | +0.086 |
| 7 | L3  | H5 | +0.358 | 0.979 | 0.949 | +0.030 |
| 8 | L7  | H0 | +0.354 | 1.845 | 1.816 | +0.030 |
| 9 | L15 | H2 | +0.347 | 1.692 | 1.626 | +0.066 |
| 10| L3  | H1 | +0.342 | 0.663 | 0.636 | +0.028 |

**Read:** Top d ≈ 0.56, very similar to Gemma. The notable structural difference is **L23 clustering**: three of the top-5 heads (H0, H1, H7) are in L23, the final attention layer. Layer 23 is the most downstream attention before the output head, so late-layer commitment is a plausible mechanistic interpretation. Top-10 covers 21% of all Qwen3.5 attention heads — denser than Gemma's 3.6% because there are far fewer heads to distribute across.

## Cross-model comparison

|  | Gemma 4 E2B | Qwen3.5-0.8B |
|---|---|---|
| Total attention heads | 280 (35 layers × 8) | 48 (6 layers × 8) |
| Cohort sizes (cr / cw) | 53 / 38 | 64 / 18 |
| Top Cohen's d | 0.599 | 0.557 |
| # heads with d ≥ 0.5 | 2 | 3 |
| # heads with d ≥ 0.4 | 4 | 5 |
| Top-signal structure | Distributed across 7 layers | **L23 has 3 of top-5** |
| Signal magnitude (mean Δ of top-5) | +0.320 | +0.101 |

## Interpretation

### 1. The overconfidence signal is real but modest in both models

In both models the top head has Cohen's d ≈ 0.6 — a medium effect in statistical terms. This is large enough to be detected at the current cohort sizes but small enough that per-head intervention effects in Phase B.2 are likely to be partial, not dramatic. **We should not expect tempering to eliminate hallucination. We should expect it to reduce hallucination rate by some measurable fraction if the attribution is causal.**

### 2. No single head carries the behavior

Neither model has a single head with d > 1 or Δ dominating by orders of magnitude. This directly mirrors the `FINDING_pca_rank.md` result on chef_dpo: the DD→base shift is mid-rank and distributed, not a sharp attractor. **Attention-head overconfidence is a distributed signal, not a single-circuit phenomenon.** This has two implications:

- A single-head tempering experiment is unlikely to suffice. Top-K tempering with K ≥ 5 is the minimum plausible intervention.
- A CCA-style multi-direction intervention remains a viable alternative path if top-K tempering underperforms.

### 3. Qwen3.5's L23 clustering is the most interesting structural result

Three of Qwen3.5's top-5 heads are in layer 23, the final attention layer. Late-layer concentration is consistent with the mechanistic hypothesis that overconfident delivery is a *commitment* signal expressed close to the output — the model "decides" to deliver confidently after content has been composed in earlier layers. If Phase B.2 confirms that L23 heads are causally involved in Qwen3.5's overconfidence, that is a cleaner mechanistic story than Gemma's distributed signal.

Note that this clustering could equally be an artifact of Qwen3.5's hybrid architecture — with only 6 attention layers, late-layer dominance may reflect information aggregation from 18 upstream DeltaNet layers rather than a disposition-specific circuit. Phase B.2 would disambiguate.

### 4. Raw activation differences are tiny in magnitude

Δ values are on the order of 0.03–0.48 on base norms of 1–16. The largest relative difference is ~8% (Gemma L16-H5). Cohen's d is detectable because per-sample variance is small; the mean shift itself is small in absolute terms. **This tells us the overconfidence signature is a subtle perturbation on top of normal attention activity, not a wholesale redirection of attention.** Tempering with λ = 0.5 may be too aggressive; a finer λ sweep starting at 0.9 may be more appropriate.

### 5. Gemma remains the stronger Phase B.2 target

On pure cohort size, Gemma's cw=38 gives meaningfully more statistical power than Qwen3.5's cw=18. The Cohen's d values are comparable between models, but the Qwen3.5 signal should be interpreted as "indicative at n=18" rather than "confirmed." Running B.2 on Gemma first is the right call; a Qwen3.5 B.2 replication can follow if the Gemma result is positive.

## Phase B.2 candidate heads (from this attribution)

**Gemma 4 E2B — top-5 to temper:**
1. L16-H5 (d=+0.599)
2. L20-H7 (d=+0.597)
3. L3-H3 (d=+0.437) — note earliest layer in top-5
4. L18-H5 (d=+0.431)
5. L17-H0 (d=+0.381)

**Qwen3.5-0.8B — top-5 to temper (for a later follow-up):**
1. L7-H6 (d=+0.557)
2. L23-H1 (d=+0.544)
3. L19-H2 (d=+0.500)
4. L23-H0 (d=+0.469)
5. L23-H7 (d=+0.442)

## Caveats

1. **Correlational, not causal.** A head with high Cohen's d may fire more on cw either because it *causes* overconfident delivery, or because it *responds to* content that is hard for the model (the kind of content cw happens to cover). Only Phase B.2 tempering can disambiguate.

2. **Qwen3.5 cw=18 is tight.** The d ≈ 0.5 effects are detectable at this n but have wide confidence intervals. Results should be treated as candidate identification, not confirmed attribution.

3. **Last-32-token pooling is an arbitrary choice.** The plan specifies "last few generated tokens" as where delivery commitment is expressed. 32 tokens is a reasonable default but has not been validated — earlier (prompt-adjacent) tokens or shorter pools might produce different rankings. Ablation on pool length is cheap and worth running if B.2 results are ambiguous.

4. **Activation magnitude, not direction.** We measure L2 norms, not dot products or directional alignment. A head that activates with the same magnitude but different direction on cw vs cr would be invisible to this attribution. A directional version of the attribution (mean vector per cohort, cosine distance between cw-mean and cr-mean) is a natural follow-up if this magnitude attribution underpredicts the B.2 tempering effect.

5. **Gemma 4's gated attention architecture.** The o_proj input dimension matches the standard `n_heads × head_dim = 8 × 256 = 2048` at runtime, consistent with a conventional multi-head attention path. Unsloth's warning ("Gemma4 does not support SDPA — switching to fast eager") does not affect the attribution math; it only affects which kernel runs under the hood.

6. **Hybrid Qwen3.5 attention-only attribution.** 18 DeltaNet layers are silently excluded. If overconfidence is partly expressed in DeltaNet state rather than attention, this method will miss it. A DeltaNet-specific attribution is possible but requires a different per-unit measurement scheme.

## What this adds to the plan

- **Phase B.1 is complete for both Gemma 4 E2B and Qwen3.5-0.8B.** Both have ranked head lists and candidate top-5 sets for tempering.
- **The attribution signal exists at a measurable but modest scale.** Large enough to proceed to B.2, small enough to predict partial rather than dramatic tempering effects.
- **Gemma is the stronger B.2 target** on cohort-size grounds.

## Files

- `dd-v2/eval/path_b_attribution.py` — attribution script (Gemma + Qwen3.5 compatible)
- `dd-v2/eval/results/path_b_attribution_gemma4.json` — full Gemma head ranking + per-layer stats
- `dd-v2/eval/results/path_b_attribution_qwen35_0.8b.json` — full Qwen3.5 head ranking + per-layer stats
- `dd-v2/eval/logs/path_b_gemma4.log` — run log
- `dd-v2/eval/logs/path_b_qwen35.log` — run log
