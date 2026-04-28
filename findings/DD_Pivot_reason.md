# DD Pivot Reason — Retrospective of Phase 0, Phase 1, and DD v2

**Date:** 2026-04-09
**Status:** Consolidated — all three research arcs of Disposition Distillation have closed negative at sub-2B scale.
**Scope:** Everything from original hypothesis through DD v2 Path B falsification.
**Purpose:** Single document capturing why the DD program as originally framed is being retired, what was tried, what the numbers were, and what remains open.

---

## 1. Original hypothesis (going into Phase 0)

Disposition Distillation (DD) would train **behavioral tendencies** (not task performance) into sub-billion-parameter LLMs via a 4-stage all-MIT multi-teacher SFT pipeline (Kimi K2.5 / GLM-5 / MiniMax M2.7 / Synthesizer) + LoRA. Seven target dispositions: Eager, Deliberate, Adversarial, Curious, Self-Improving, Humble, Persistent.

Success criteria:
- **HumanEval delta ≥ 0** (non-regression)
- **MCAS delta > +5 pts** (disposition transfer)
- **Quantized MCAS preserved vs baseline ≥ +10 pts**
- **RYS layer-duplication** amplifies disposition (DD+RYS / DD-only ≥ 1.15× on SVR)

Compound stack: BitNet 1.58-bit + APOLLO-Mini + TurboQuant KV + RYS + DD LoRA + semantic-memory cache.

---

## 2. Phase 0 — DD SFT on Qwen3-0.6B (2026-03-28 to 2026-04-05)

### 2.1 Pipeline and training
- 1,250 prompts → 4-stage pipeline → 1,085 filtered (86.8%)
- LoRA r=64 α=128, **all-modules** (q/k/v/o + gate/up/down), APOLLO-Mini rank-1 scale=128
- 3 epochs, batch 32, ctx 8192, 2.6 min on RTX 4090, loss 1.835 → 1.214
- Cost: ~$47.50 pipeline + ~$7.50 eval

### 2.2 Published "positive" Phase 0 results (later invalidated)
Eval: llama-server `/completion`, temp=0, **n_predict=512**.

| Metric | Baseline | DD FP16 | Δ |
|---|---|---|---|
| HumanEval Pass@1 | 14.6% (24/164) | 29.9% (49/164) | **+15.3** |
| MCAS (Claude Sonnet judge, 500 prompts) | 53.1% | 87.0% | **+33.9** |
| SVR | 95.8% | 95.4% | −0.4 |
| Uncertainty Discrimination | −75.9 | −4.6 | +71.3 |

Single-teacher ablation: GLM-5 only → HumanEval 2.4% (worse with more data). Pipeline's +27.5 pts "incremental value" claim rested on this comparison.

### 2.3 Quantization (Phase 0c) — the one robust finding
llama.cpp GGUF, true greedy:

| Format | Size | HumanEval | MCAS | SVR |
|---|---|---|---|---|
| FP16 | 1.14 GB | 29.9% | 87.0% | 95.4% |
| Q5_K_M | 418 MB | 25.0% | 84.3% | 88.0% |
| Q4_K_M | 373 MB | 25.0% | 73.6% | 88.0% |
| Q2_K | ~200 MB | — | — | catastrophic |

- BitNet TQ2_0 / TQ1_0: Unicode garbage at 0.6B. Ternary fails below ~4 BPW.
- TurboQuant: requires head_dim ≥ 128; Qwen3 head_dim=64. **Not viable** → use Q8_0 KV (1.3× savings, no quality loss).

### 2.4 RYS layer duplication (Phase 0.5) — FAILED
Disposition cortex identified at layers 22–26 (activation MAE peaks at L26 = 0.263).

| Config | Layers | Sonnet MCAS | Kimi MCAS | SVR |
|---|---|---|---|---|
| DD no-RYS | 28 | 83.0% | 75.0% | 98.0% |
| DD + L26 dup | 29 | 65.0% | 64.0% | 94.0% |
| DD + L25,26 dup | 30 | 55.0% | 45.0% | 88.0% |
| DD + L22–26 dup | 33 | ~55% | 44.0% | 86.0% |

Full 2-layer MCAS eval (500 prompts): **5.9% MCAS, −81.1 pts**. 447/500 zero scores.
- SVR gate: PASS (1.188× > 1.15 target).
- MCAS gate: **FAIL** — pattern amplification at catastrophic quality cost.
- **Disposition cortex is dual-use at 0.6B.** RYS retired.

### 2.5 The measurement artifact (2026-04-05)
Phase 1 re-eval with `n_predict=4096` exposed that Phase 0 had strangled baseline more than DD (baseline's longer think blocks truncated at 512).

| | Phase 0 (n=512) | Corrected (n=4096) |
|---|---|---|
| Baseline Qwen3-0.6B | 14.6% | **36.0%** |
| DD all-modules | 29.9% | **28.0%** |
| Δ | +15.3 | **−8.0** |

The DD score was stable (~28–30% everywhere). The +15.3 delta was an artifact.

### 2.6 Per-variant re-eval (all LoRA configs)

| Model | LoRA | Data | HumanEval | Δ |
|---|---|---|---|---|
| Baseline | — | — | 36.0% | — |
| DD all-modules | q/k/v/o + MLP | 1,085 naive | 28.0% | −8.0 |
| DD attn-only | q/k/v/o | 1,085 naive | 33.5% | −2.5 |
| DD attn-only + SA | q/k/v/o | 1,656 student-aware | 35.4% | −0.6 |

5.5 pts of the 8-point drop came from MLP disruption. Student-aware data recovered another 1.9 pts. Best variant still regressed.

### 2.7 Apples-to-apples MCAS re-eval (Grok 4.20 judge, 164 prompts)
DD-SA 0.6B vs baseline, same judge same prompts:

| Dimension | Baseline | DD-SA | Δ |
|---|---|---|---|
| uncertainty_calibration | 2.11 | 2.11 | 0.00 |
| self_verification | 2.53 | 2.06 | −0.47 |
| error_detection | 3.11 | 2.98 | −0.13 |
| assumption_surfacing | 2.51 | 2.66 | +0.15 |
| edge_case_awareness | 2.13 | 2.26 | +0.13 |
| **mcas_total** | **12.39** | **12.07** | **−0.32** |

**The +33.9 MCAS claim did not replicate** on the best-preserving variant under an independent judge.

### 2.8 Recovery experiment (Self-Improving disposition)
Multi-turn error recovery on 164 HumanEval problems:

| Model | Pass@1 | Failures | Recovered | Recovery Rate |
|---|---|---|---|---|
| Baseline 0.6B | 36.0% | 105 | 11 | **10.5%** |
| DD-SA 0.6B | 35.4% | 106 | 8 | **7.5%** |
| Baseline 1.7B | 71.3% | 47 | 8 | **17.0%** |

**DD model recovers worse than baseline.** "Self-Improving" disposition is performative.

### 2.9 Activation steering (contrastive vectors, layers 22/24)
| Layer | α | Rate | Δ |
|---|---|---|---|
| 22 | 1.0 | 37.8% | +1.8 |
| 22 | 2.0 | 35.4% | −0.6 |
| 22 | 3.0 | 37.8% | +1.8 |
| 22 | 5.0 | 32.9% | −3.1 |
| 22 | 12.0 | 33.5% | −2.5 |
| 24 | 1.0 | 37.2% | +1.2 |

Noise at low α, degradation at high α, no dose-response. **Failed.**

### 2.10 Phase 0 verdict
- HumanEval +15.3: **ARTIFACT** (corrected −8.0)
- MCAS +33.9: **NOT REPLICATED** under independent judge (−0.32)
- Self-Improving disposition: **PERFORMATIVE** (recovery rate worse than baseline)
- RYS: **DUAL-USE CORTEX**, not viable at 0.6B
- Steering: **NOISE**
- Single robust finding: Q5_K_M quantization preserves DD-style at sub-500 MB.

---

## 3. Phase 1 — DD SFT on Qwen3-1.7B (2026-04-01 to 2026-04-04)

### 3.1 What Phase 1 tried to fix
Phase 0 "worked" because the 0.6B baseline was weak. On the stronger 1.7B (HumanEval 83.5% native), the pipeline catastrophically regressed.

### 3.2 All SFT variants

| Variant | Config | Data | HumanEval | Δ | Think retention |
|---|---|---|---|---|---|
| Baseline Qwen3-1.7B | — | — | **83.5%** | — | 1.00 (6970 char median) |
| Naive DD all-mod r=64 | 7 modules, 697 ex | Clean | 48.8% | −14.0 | — |
| Naive DD all-mod r=128 | same | Clean | 49.4% | −13.4 | — |
| Naive DD attn r=64 | q/k/v/o | Clean | 57.3% | −5.5 | — |
| Naive DD attn r=32 | q/k/v/o | Clean | 59.1% | −3.7 | — |
| V2 attn (SA, post-fix) | r=64, 3 ep | 1,656 SA | 67.7% | −15.8 | 0.48 (3331 char) |
| V2 all-modules | r=64, 3 ep | 1,656 SA | 61.6% | −21.9 | — |
| V3 rsLoRA + think-mask | r=64, 0.125× scale | 1,656 SA | ~51.2% | **−32.3** | — |
| V4 low-rank low-epoch | r=32, 1 ep | 1,656 SA | 62.8% | −20.7 | 0.58 |
| **V5 length-augmented** | r=64, 3 ep, attn | 1,656 aug to 7.3k | **76.8%** | **−6.7** | 0.96 |
| V5 + code_block prompt | (prompt fix only) | same | **81.1%** | **−2.4** | — |

V3 "mask think blocks from loss" hypothesis destroyed: **the disposition is in the thinking**; masking removes the signal.

### 3.3 The reasoning compression mechanism
Bucketed pass rate by think-block retention ratio:

| Retention | n | Pass rate |
|---|---|---|
| <0.25 | 12 | 33.3% |
| 0.25–0.50 | 54 | 66.7% |
| 0.50–0.75 | 48 | 72.9% |
| 1.00–1.50 | 15 | 73.3% |
| ≥1.50 | 10 | **90.0%** |

SFT pulls think depth to match training distribution (2,941 char training median vs 6,970 char native). Length-augmentation (V5) injected DD disposition sentences into baseline-depth reasoning scaffolds → retention jumped to 0.96, gap closed 58%.

### 3.4 Training data format mismatch
V2 training data characteristics:
- Median response 3,434 char (HumanEval expected ~200–500)
- 87% start with text (not code)
- 86% markdown headers
- 89% >2,000 chars
- 81% self-verification language
- 77% edge case discussion

Output format is **not** the damage vector — V2 on HumanEval produces cleaner output than baseline (0% markdown vs 26%). **Reasoning compression alone is the damage.**

### 3.5 V5 MCAS/SVR on Qwen3-1.7B (Grok 4.20 judge, 164 prompts)

MCAS (1–5 per disposition):
| Disposition | V5 | Baseline | Δ |
|---|---|---|---|
| Humble | 2.01 | 1.97 | +0.04 |
| Deliberate | 3.20 | 3.18 | +0.02 |
| Adversarial | 2.73 | 2.60 | +0.13 |
| **Total** | **7.94** | **7.75** | **+0.19** |

SVR (pattern counts):
| Disposition | V5 | Baseline | Δ | % |
|---|---|---|---|---|
| Humble | 0.41 | 0.36 | +0.05 | +14% |
| Deliberate | 3.40 | 3.27 | +0.13 | +4% |
| Adversarial | 1.71 | 1.53 | +0.18 | +12% |
| **Total** | **5.52** | **5.16** | **+0.36** | +7% |

**The Level 1 → Level 2 gap:** SVR (disposition language) +7%; MCAS (disposition behavior) +2.4%. **3:1 language-to-behavior ratio.** V5 regressions include 25k-char think blocks with 4+ disposition patterns that still fail the problem. Model says "let me verify" but does not verify.

### 3.6 DPO attempt (Phase 1)
DPO with preference pairs was attempted on Qwen3-1.7B and **also failed** — same style-transfer-not-substance pattern. (This is why it was not included in DD v2's pivot options.)

### 3.7 Phase 1 verdict
**SFT teaches the model to say what a careful reasoner says, not to be one.** Every SFT configuration tried — naive, attn-only, all-modules, think-masked, low-rank, length-augmented — regressed on HumanEval. Best variant V5+prompt still −2.4 pts, with only +0.19 MCAS. DPO failed the same way. **SFT is the wrong paradigm.**

---

## 4. DD v2 — French Chef specialist + inference-time Path B (2026-04-06 to 2026-04-08)

### 4.1 Reframe
Accept that SFT regresses. Pivot to an **inference-time, ground-truth-anchored** claim:

> Disposition is a low-rank, inference-time-steerable axis separable from content. Each of 7 dispositions can be addressed by damping a small set of attention heads (~5) identified via activation-based attribution on a confident-wrong vs confident-right cohort split. Artifact: `base_model → {disposition: [(layer, head, λ)]}` composed at inference, no per-specialist retraining.

**Dual criterion:** (a) disposition metric shifts in target direction; (b) gold-checklist coverage preserved.

### 4.2 Chef SFT+DPO as sanity ceiling (Step 16, Gemma 4 E2B base)
Before Path B, the training-side pipeline was rerun on a new base with clean data (684 prompts, boundary cases, 4-teacher pipeline with DeepSeek V3.2 swapped in for Stage 2).

Gold checklist (100 French cuisine prompts):
| Condition | Coverage |
|---|---|
| Baseline Gemma 4 E2B | **0.452** |
| Single-teacher SFT | 0.420 |
| Multi-teacher SFT | 0.415 |
| SFT only (T1+T2+T4) | 0.413 |
| **Full DD v2 (SFT+DPO)** | **0.379** (−7.3) |
| Full DD v2 + sysprompt | 0.368 (−8.4) |

Rubric judge MCAS proxy: 4.40 → 2.50 (−1.90). DeepSeek head-to-head: DD 45% vs baseline 52%.

**Novel-prompt probe:** DD v2 Chef invents fabricated techniques ("Steam-Sear," "Inclusion Mask," "Visual Overload") in a McKinsey consulting voice. **Voice perfectly imitated, content catastrophically absent.** Training data is clean at the example level; the loss is in the weights. Sysprompt cannot recover it.

**Chef imitation pipeline falsified** — the same pattern as Phase 0 and Phase 1, now on a 3rd base model on a 3rd domain. Uniform stylistic teacher distribution causes sub-2B SFT to compress toward style features at the cost of content recall.

### 4.3 Path B — Baseline disposition measurement (cross-model)
100 chef prompts, 2×2 confident-right / confident-wrong / hedged-right / hedged-wrong:

| Model | cr | hr | cw | hw | Acc | Assert asymmetry | Hall rate | Coverage |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 E2B (greedy) | 53 | 5 | 38 | 4 | 0.58 | **−0.009** | 90.5% | 0.452 |
| Qwen3.5-0.8B (sampling) | 64 | 11 | 18 | 7 | 0.75 | **−0.133** | 72.0% | 0.219 |
| SmolLM2-1.7B (greedy) | 45 | 11 | 33 | 11 | 0.56 | −0.054 | 75.0% | 0.324 |

Assertion is **statistically independent of correctness** in Gemma and SmolLM. Only Qwen3.5 shows weak real calibration signal. Confident hallucination is universal: 72–91%.

### 4.4 Path B.1 — Magnitude attribution (Cohen's d on per-head L2 norms, last 32 tokens)

**Gemma 4 E2B Humble** (cr=53, cw=38):
- Top-5: L16-H5 (d=0.599), L20-H7 (0.597), L3-H3 (0.437), L18-H5 (0.431), L17-H0 (0.381)
- Distributed across 7 layers (3, 16, 17, 18, 20, 27, 30). Max raw Δ ~8% of per-head norm.

**Gemma 4 E2B Adversarial-self** (self_verif cohort split, advH=11, advL=69):
- Max |d| = **0.875** at L23-H0 — strongest signal in the project
- **L11 cluster** (H0, H5, H6) in top-10
- Top-5: L23-H0, L11-H5, L11-H0, L11-H6, L15-H4 (all negative d → damping should raise self-verification)

**Qwen3.5-0.8B Humble** (cr=64, cw=18, only 6 attention layers; 18 layers are DeltaNet):
- Top-5: L7-H6 (0.557), L23-H1 (0.544), L19-H2 (0.500), L23-H0 (0.469), L23-H7 (0.442)
- **L23 cluster: 3 of top-5 heads in final attention layer.** Structurally different from Gemma's distributed signal.

**Deliberate** (pedagogical_framing axis): 98/100 rubric-saturated high. **Not attributable** on chef — no cohort split possible.
**Curious** (certainty axis): 0/100 marked AMBIGUOUS. **Not attributable.**

### 4.5 Path B.2 — Tempering sweep (forward_pre_hook on `self_attn.o_proj` input, per-head slice × λ)

**Gemma 4 E2B Humble** (full sweep, greedy):
| λ | cr | hr | cw | hw | Acc | Assert | Hall | Cov |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 60 | 6 | 29 | 5 | 0.66 | 0.89 | 0.853 | 0.489 |
| 0.9 | 52 | 7 | 36 | 5 | 0.59 | 0.88 | 0.878 | 0.492 |
| 0.8 | 52 | 6 | 35 | 7 | 0.58 | 0.87 | 0.833 | 0.470 |
| 0.5 | 54 | 5 | 33 | 8 | 0.59 | 0.87 | 0.805 | 0.486 |

Assertion frozen (0.89→0.87). Accuracy dropped 7 pts. cw moved **wrong direction**. **Null on disposition, modest content damage.**

**Gemma 4 E2B Adversarial-self** (killed after λ=0.9, pattern confirmed):
| λ | sv_mean | high(≥4) | low(≤2) | Cov |
|---|---|---|---|---|
| 1.0 | 3.69 | 82 | 17 | 0.474 |
| 0.9 | 3.58 | 75 | 23 | 0.479 |

Self-verification moved **wrong direction** (3.69 → 3.58) despite strongest localized signal in project. Coverage flat. Killed at λ=0.9.

**Qwen3.5-0.8B Humble** (main run, seed 1000+i, sampling t=1.0/tp=1.0/tk=20, enable_thinking=False):
| λ | cr | hr | cw | hw | Acc | Assert | Hall |
|---|---|---|---|---|---|---|---|
| 1.0 | 19 | 2 | 72 | 7 | 0.210 | 0.910 | 0.911 |
| 0.9 | 14 | 3 | 75 | 8 | 0.170 | 0.890 | 0.904 |
| 0.8 | 12 | 3 | 75 | 10 | 0.150 | 0.870 | 0.882 |
| 0.5 | **22** | 1 | **63** | 14 | **0.230** | 0.850 | **0.818** |

λ=0.9/0.8 content damage (cr monotonic drop, cw flat). λ=0.5 **apparent positive**: cr recovered +3, cw −9, hall −9.3 pts, hw +4 (cw→hw migration). First apparent clean signal in the project. **Non-monotonic path flagged as red flag.**

**Qwen3.5-0.8B Humble REPLICATION** (seed 2000+i, same everything else):
| | cr | hr | cw | hw | Acc | Assert | Hall |
|---|---|---|---|---|---|---|---|
| Main λ=0.5 | 22 | 1 | 63 | 14 | 0.230 | 0.850 | 0.818 |
| **Rep λ=0.5** | **12** | 1 | **77** | 10 | **0.130** | 0.890 | **0.885** |

Hit every pre-declared sampling-artifact criterion. **Replication failed.** Main λ=0.5 was an RNG draw, not a real effect.

### 4.6 Path B.1 — Directional attribution (Qwen3.5 Humble, final methodological check)
Captures full per-head activation vectors per sample. Direction v = µ_cw − µ_cr. Projects samples onto v. Cohen's d on projections.

- Top-10 d_directional: 1.73, 1.58, 1.49, 1.47, 1.43, 1.41, 1.40, 1.31, 1.30, 1.29
- **cos(µ_cr, µ_cw) on every top head ≈ 1.0** (0.958–0.998)
- Cohort means nearly collinear → large d is a variance artifact (projecting onto near-zero orthogonal component, pooled std → 0, d inflates)
- Ranks same L11/L19/L23 cluster as magnitude — **not a hidden axis**
- cr and cw cohorts are **not linearly separable** in per-head activation space, magnitude or directional.

### 4.7 PCA rank of DD−base diff (layer −8 residual, 80 prompts, 1536-d)
| | dims at 90% EVR | top PC EVR |
|---|---|---|
| Base | 32 | — |
| DD | 32 | — |
| **Diff (DD−base)** | **47** | **0.110** |

DD diff is **mid-rank and distributed** (47 dims at 90% variance, 47% more spread than base's 32). Top PC only 11%. **No low-rank attractor.** The core DD v2 assumption "disposition is a low-rank axis" is empirically false at this scale.

### 4.8 Path B consolidated verdict (5 experimental variants)
| # | Model | Disposition | Method | Disposition moves? | Content preserved? |
|---|---|---|---|---|---|
| 1 | Gemma 4 E2B | Humble | magnitude | ❌ frozen | ❌ 7 pt drop |
| 2 | Gemma 4 E2B | Adv-self | magnitude | ❌ wrong direction | ✅ flat |
| 3 | Qwen3.5-0.8B | Humble | magnitude | ⚠️ non-monotonic | ⚠️ RNG-dominated |
| 4 | Qwen3.5-0.8B | Humble (rep) | magnitude | ❌ wrong direction | ❌ worse |
| 5 | Qwen3.5-0.8B | Humble | directional | N/A — no separable axis | N/A |

**Zero clean positives.** The narrow Path B claim is falsified: forward-pass cohort attribution (magnitude or directional) + scalar head damping does not produce reliable disposition shift at sub-2.4B scale, on either tested architecture, on either tested disposition, with either tested attribution method.

---

## 5. The pattern across all three arcs

Across Phase 0 (0.6B Qwen3), Phase 1 (1.7B Qwen3), and DD v2 (Gemma 4 E2B + Qwen3.5-0.8B), **every primary DD intervention regressed**:

| Arc | Method | Base | Result |
|---|---|---|---|
| Phase 0 | SFT all-modules LoRA | Qwen3-0.6B | HumanEval −8.0 (after artifact correction), MCAS −0.32 (apples-to-apples), recovery rate −3.0 pts |
| Phase 0.5 | RYS layer duplication | DD Qwen3-0.6B | MCAS −81.1 pts (SVR gate passed at catastrophic quality cost) |
| Phase 0 | Activation steering | DD Qwen3-0.6B | Noise at low α, degradation at high α |
| Phase 1 | SFT attn-only, V2-V5, rsLoRA, think-mask | Qwen3-1.7B | Best V5+prompt −2.4 pts HumanEval, MCAS +0.19 (noise), 3:1 language-to-behavior ratio |
| Phase 1 | DPO | Qwen3-1.7B | Failed (same style-transfer pattern) |
| DD v2 | SFT + DPO Chef imitation | Gemma 4 E2B | Gold coverage −7.3 pts, rubric MCAS −1.90, novel prompts confabulate in McKinsey voice |
| DD v2 | Path B magnitude + directional | Gemma + Qwen3.5 | 5 variants, zero clean positives |

**Root mechanism (converged from all three arcs):**
1. Base MLPs store content; attention routes it.
2. Teacher-distilled training data has uniform style and diverse content.
3. Sub-2B SFT compresses toward the high-uniformity style signal at the cost of non-uniform content recall.
4. The result is a model that **sounds** like the target disposition but does not **execute** it — Level 1 transfer without Level 2.
5. Inference-time interventions fare no better: the disposition signal in attention is not low-rank, not linearly separable, and not causally isolatable by scalar head damping at this scale.

**Base model's disposition is already present and promptable.** The Gemma Chef sysprompt experiment recovered verifier keyword density via prompting alone, while DD-trained weights could not be recovered even with anti-deflection prompts. DD is anti-flexibility, not pro-disposition.

---

## 6. What is falsified vs what is not

### Falsified (at sub-2.4B scale, on tested dispositions/bases)
- DD via SFT (any LoRA config, any data length, any think-mask, any rank/epoch)
- DD via DPO (Phase 1 Qwen3-1.7B)
- RYS layer duplication as disposition amplifier
- Activation steering from contrastive DD vectors
- Path B scalar head damping via magnitude attribution
- Path B scalar head damping via directional attribution
- The "disposition is a low-rank axis" prior (PCA rank 47 at 90% variance)

### Not falsified
1. **Disposition at larger scales (~7B+).** All tested models sub-2.4B. Circuit localization may increase with scale.
2. **Other intervention classes.** v_proj, MLP down_proj, residual-stream targeting, inference-time low-rank adapters, gradient-based attribution, activation patching — untested.
3. **Other dispositions.** Curious, Persistent, Eager, Self-Improving untested on Path B. Adversarial-self partially tested on Gemma only. Deliberate not attributable (rubric saturation).
4. **Execution-rich training data.** All SFT data was essays/explanations. True execution traces (actual errors, actual fixes, actual re-runs) never tested.
5. **Quantization preserves DD-style content** (Q5_K_M validated Phase 0c). This is the one robust positive.

---

## 7. Pivot rationale

The narrow "simple recipe, same method everywhere" version of Disposition Distillation is not viable as the research program's center of gravity. Three honest reframes, plus the chosen path:

**Reframe A — Model-scale hypothesis.** Test Path B at 4–7B+. Cost ~$20–40 per base. Defensible but expensive and speculative given the strength of the sub-2B negative.

**Reframe B — Intervention-class hypothesis.** Scalar head damping is the wrong primitive. Try low-rank inference-time adapters, residual stream targeting, or activation patching. Significant engineering, plausibly effective.

**Reframe C — Disposition-is-not-mechanistic hypothesis.** Disposition is an emergent property of the full forward pass, shaped by training data distribution, not an identifiable feature direction. Training-level interventions are the only lever — and those have been shown performative-only at sub-2B.

**Reframe D (chosen) — Accept falsification at this scale, pivot research direction.** Document the cross-arc cross-model negative as the main DD result. Reframe the program around a sharper, narrower, publishable thesis:

> Disposition distillation at sub-2B is not a separable low-rank axis under the interventions we tested; the shaping tools that work are (a) full training or (b) prompt engineering, both of which have scale-specific limits. Quantization preserves style signal but not functional disposition. Model size matters: circuits may not be localized at <2.4B.

This is a real, publishable negative that contributes to the interpretability and alignment literature by narrowing the space of viable claims. Specifically:
- Rules out cheap inference-time recipes at sub-2B.
- Rules out SFT-based disposition transfer on strong-baseline models.
- Validates quantization (Q5_K_M) as the single robust stack element.
- Quantifies the Level 1 / Level 2 language-behavior gap (3:1 ratio in V5).
- Provides the first cross-model, cross-architecture, cross-method null for scalar head tempering.

---

## 8. Decisions recorded (2026-04-08 / 2026-04-09)

1. **Path B closed** as primary DD v2 method. No further B.1/B.2 sweeps on currently tested models.
2. **Chef imitation pipeline closed.** Novel-prompt probe is sufficient falsification.
3. **SFT-based DD closed** across all tested bases (0.6B, 1.7B, 2.3B effective) and all tested data configurations.
4. **DPO-based DD closed** (Phase 1 already failed).
5. **RYS, steering, TurboQuant closed.**
6. **Q5_K_M + Q8_0 KV cache retained** as the one robust deployment finding.
7. **Pivot direction:** write the cross-arc negative as the main contribution. Reframe the research program around narrower claims.

---

## 9. Cost accounting across all three arcs

| Arc | Compute | API | Total |
|---|---|---|---|
| Phase 0 (pipeline + training + eval + Phase 0.5) | ~$60 | ~$55 | ~$115 |
| Phase 1 (V2–V5 SFT + MCAS/SVR) | ~$15 | ~$18 | ~$33 |
| DD v2 Chef (pipeline + SFT+DPO + Step 16) | ~$25 | ~$20 | ~$45 |
| DD v2 Path B (B.1 attribution + B.2 sweeps + replication + directional) | ~$20 | ~$5 | ~$25 |
| **Total DD program** | **~$120** | **~$98** | **~$218** |

Materially under the ~$545–735 Phase 0–6 budget projected in CLAUDE.md — because the gating criteria functioned correctly and halted the program before the expensive Phase 2–6 spend.

---

## 10. Files (authoritative source list)

**Phase 0:**
- `phase0/DD_FINDINGS_COMPREHENSIVE.md`
- `phase0/ALERT_humaneval_reeval.md`
- `phase0/REEVAL_SESSION_2026_04_05.md`
- `phase0/PHASE0_RESULTS.md`
- `phase0/eval/phase0v2d_results.json`, `rys_mcas_results.json`, `recovery_*.json`, `ablation_single_teacher_results.json`, `confidence_alignment_results.json`

**Phase 1:**
- `phase1/` findings docs (V2/V3/V4/V5 sweep logs, MCAS/SVR Grok evals, training configs)

**DD v2:**
- `dd-v2/PLAN.md`, `dd-v2/SESSION_2026_04_07.md`
- `dd-v2/findings/CORE_CLAIM_ddv2.md`
- `dd-v2/findings/FINDING_crossmodel_baseline_disposition.md`, `FINDING_gemma4_baseline_disposition.md`
- `dd-v2/findings/FINDING_path_b1_attribution.md`
- `dd-v2/findings/FINDING_path_b2_gemma4_negative.md`, `FINDING_path_b2_qwen35_humble.md`
- `dd-v2/findings/FINDING_pca_rank.md`
- `dd-v2/findings/FINDING_path_b_falsified.md` (consolidated)
- `dd-v2/findings/ARCHIVE_chef_dpo_lessons.md`
- `dd-v2/eval/results/path_b_attribution_*.json`, `path_b2_*_summary.json`, `baseline_*_v2_disposition_profile.json`, `pca_rank.json`

---

**Bottom line:** Three independent research arcs, three different base models, three different intervention paradigms (SFT, DPO, inference-time attribution), one consistent result: at sub-2.4B scale, disposition is not a low-rank, separable, trainable-and-preserving axis. The base model already has it; the interventions we have at this scale cannot shape it without breaking content recall. The pivot is to document this rigorously as the main DD finding and narrow the program to claims that sub-2B evidence actually supports.

---

## 11. Open branch: frozen-base disposition adapters (not yet tested)

Every DD arc documented above shares one property: **the base model's weights moved**. Phase 0 SFT, Phase 1 SFT/DPO, and DD v2 Chef SFT+DPO all overwrote base parameters. DD v2 Path B did not train, but it still treated base's own attention routing as the surface to modify (via scalar damping). Every failure observed — content regression in SFT arcs, RNG-dominated nulls in Path B — is downstream of the same choice: the operator acted on base weights or base activations directly.

A branch that was never tested, and that is explicitly motivated by the failure mode of the above, is:

> **Frozen-base disposition adapters.** Base model weights are never touched. A separate, small, trainable module sits at or after the base's output and applies the disposition transformation. The adapter either rewrites the base's output post-hoc or modifies the decode-time logit stream.

**This does not invalidate the DD negative documented in §§ 2–10.** It represents a *different operator on a different surface*. Imitation-into-weights is falsified. A post-hoc, frozen-base, reward-trained adapter has not been tested at all. The prior from the sub-2B negative is that *weight-modifying* operators corrupt; it says nothing about adapters that cannot corrupt by construction.

### 11.1 Why this branch structurally avoids the failure mode

1. **Corruption is impossible.** Base HumanEval is 83.5% before and 83.5% after, by construction. The "we corrupt the natural responses with even 1,656 examples" observation from Phase 1 / DD v2 is not a tuning problem to beat — it's a property of distribution-matching losses on base weights. A frozen base cannot be corrupted by any training signal.

2. **The learning problem is bounded.** Instead of "distill a disposition from 1,656 essays into a 1.7B model's parameters," the task becomes "given this specific base response (and optionally its uncertainty signal), emit the dispositional edit." The adapter operates on the base's own distribution as input. Much smaller, better-posed learning target.

3. **Base uncertainty is a readable input signal.** Per-token entropy, top-2 logit margin, and final hidden state are available from the base forward pass at zero extra cost. This is the first candidate signal in the entire DD program that mechanistically distinguishes confident-right from confident-wrong — exactly the failure mode we documented in §4.3 where Gemma shows assertion asymmetry of −0.009 (assertion statistically independent of correctness).

4. **Reward-in-the-loop becomes trivial.** Since base is frozen, the adapter can be trained directly with GRPO / RFT against the dual criterion (coverage × hedging-on-wrong). There is no need for a KL leash — the base *is* the anchor. This unifies the §7 "use RL not SFT" argument with a surface where RL can actually apply.

5. **Published precedent exists.** This is not speculative architecture. Aligner (Ji et al., NeurIPS 2024), Proxy-tuning (Liu et al. 2024), DExperts, and Contrastive Decoding are all published, reproduced, frozen-base instances of the same family. We would be reproducing known-working patterns, not inventing.

### 11.2 Explicit ceiling: expressible vs executable dispositions

A post-hoc or logit-level adapter can shape **expression** but not **reasoning**. If base's `<think>` block never actually traces a test case, a rewriter can add "let me verify…" language but cannot make the verification happen. This partitions the 7-disposition taxonomy:

- **Expressible (adapter is a natural fit):**
  - Humble — selective hedging on base-uncertain outputs
  - Eager — warmth in delivery
  - Curious — follow-up questions
  - Persistent — retry framing
  - Tone-level Deliberate (pacing, register)

- **Executable (adapter has a ceiling; requires reasoning change inside base):**
  - Adversarial-self — requires base to actually self-critique mid-reasoning
  - Self-Improving — requires base to change behavior after feedback (Phase 0 recovery experiment already showed this is a capability, not a learnable style)
  - Execution-level Deliberate — tracing, test-case walking

This is a feature, not a liability, for the pivot: it carves the taxonomy into what a frozen-base adapter can honestly claim and what it cannot. "Expressible" is what "disposition" means in ordinary usage anyway; the executable dispositions were always leaning on capability, not character.

### 11.3 Experiment order (cheapest-and-most-diagnostic first)

**Test 1 — Calibrated hedging head.** *First and cheapest.*
- **Architecture:** tiny 2-layer MLP, ~1M parameters.
- **Input:** base entropy, top-2 logit margin, final-token hidden state.
- **Output:** discrete action — `hedge / qualify / ask / no-change`.
- **Training:** GRPO against the dual-criterion reward (coverage × hedging-on-wrong) on Chef, base frozen.
- **Why first:** cheapest; directly targets the assertion-asymmetry = −0.009 failure; no full rewrite model needed; auditable (can read out when it fires and verify it fires on wrong answers); a clean positive here falsifies the strongest version of the §10 negative in the narrowest possible way.
- **Success criterion:** post-adapter assertion asymmetry ≤ −0.15 on Gemma 4 E2B Chef, coverage within judge noise (±2 pts) of baseline 0.452.
- **Budget estimate:** ~$10 compute.

**Test 2 — Aligner-style post-hoc editor.** *If Test 1 works.*
- **Architecture:** small encoder-decoder (~100M).
- **Training data:** `(base_response, disposition_edited_response)` pairs generated by running base on 1k Chef prompts and having a strong teacher produce the dispositional edit.
- **Why second:** most direct "always-on disposition" implementation; generalizes beyond hedging to the other expressible dispositions; transferable across base models per Aligner paper.
- **Budget estimate:** ~$15–25.

**Test 3 — Proxy-tuning / differential sidecar.** *If decode-time control is wanted over post-hoc rewriting.*
- **Architecture:** tuned small model + untuned twin, apply logit difference to frozen large base at decode time.
- **Why third:** more elegant than post-hoc rewriting; single forward pass; but requires more infrastructure and the per-token intervention is harder to audit than Test 1's discrete action or Test 2's editable text.
- **Budget estimate:** ~$20–30.

### 11.4 What a positive result on Test 1 would actually claim

> On Gemma 4 E2B Chef, a frozen base + ~1M-parameter uncertainty-gated hedging head, trained with a dual-criterion reward, can shift assertion-asymmetry from −0.009 to <−0.15 without moving coverage outside judge noise.

That is a narrow, falsifiable, frozen-base claim that:
- does not require base weights to move,
- does not require disposition to be a low-rank axis in base activations,
- does not require SFT,
- does not require a new base model,
- does not require scale beyond what we already have,
- directly addresses the one concrete baseline failure we measured and never intervened on.

It is also the smallest possible experiment that can either succeed on its own merits or, if it fails, close the *frozen-base* version of DD as decisively as §§ 2–10 closed the weight-modifying version.

### 11.5 Novelty scope — what is and is not new

Most of the ingredients in §11.1–§11.3 have nearby published precedent. It is important to be precise about which parts are reproductions of known work and which part is the actual claim, so that a future writeup is defensible and does not overclaim.

**Not novel (explicit acknowledgements):**
- Post-output rewriting as a class → Aligner (Ji et al., NeurIPS 2024)
- Decode-time logit deltas from a paired tuned/untuned twin → Proxy-tuning (Liu et al. 2024), DExperts, Contrastive Decoding
- Inserting an extra adapter path on a frozen base
- K/V memory sidecars
- Generic "knowledge sidecar" framings
- Entropy / logit-margin as an uncertainty signal

Any one of these alone would be a reproduction, not a contribution.

**Novel in combination:**
The specific combination that is *not* cleanly claimed in the adjacent literature, and that is directly motivated by the §§ 2–10 negative, is:

> **A frozen-base, always-on disposition adapter that is (1) trained on behavioral deltas rather than full response distributions, (2) gated by base-model uncertainty signals rather than applied uniformly, and (3) evaluated against domain-grounded specialist verification contracts rather than generic preference or judge scores.**

Each of those three constraints is doing load-bearing work:

1. **Delta-trained, not distribution-trained.** The whole corruption lesson from §§ 2–6 is that training on full teacher responses pulls the model toward the teacher's distribution. Training only on the *delta* between base's own output and the dispositional edit keeps the training target anchored to base behavior by construction — even before the frozen-base property kicks in. This is the part the user's "takes the output from the model and applies disposition to it" framing uniquely captures.
2. **Uncertainty-gated, not uniform.** The §4.3 cross-model baselines showed assertion asymmetry ≈ 0 — base models assert independently of correctness. A uniformly-applied disposition ("always hedge," "always qualify") would destroy calibration on the correct cases, which is the exact failure mode we saw with DD v2 Chef + sysprompt (verifier keyword density up, canonical coverage down). Gating the adapter on base's own uncertainty signal is what lets the adapter *selectively* hedge wrong answers without damaging right ones. That is the dual criterion operationalized as an architectural constraint rather than a training objective.
3. **Specialist verifier contracts, not generic judges.** The §§ 4.2–4.5 evaluations used gold-checklist coverage + dual-criterion cell counts — narrow, domain-grounded, reproducible. The adjacent alignment literature evaluates against broad preference judges and generic helpfulness scores, which is how Aligner-class work proved its point. Evaluating a disposition adapter against a *specialist verification contract* (canonical claims for Chef, citation groundedness for GDPR, execution pass for Coding) ties the adapter to measurable ground truth in a way generic judges do not, and directly addresses the §3.5 Level-1-vs-Level-2 gap (language transfer vs behavior transfer).

**The clean claim form:**

> We propose an **uncertainty-gated, delta-trained disposition adapter** for frozen base models, where a small sidecar learns behavioral edits rather than full response distributions, is gated by the base's own per-token uncertainty signal, and is evaluated under domain-grounded specialist verifiers rather than generic preference scores. The claim is motivated by a negative: imitation-into-weights DD corrupts base behavior at sub-2B scale (§§ 2–10), and the known adjacent families (Aligner, Proxy-tuning, DExperts) individually test components of but not the full combination of these three constraints.

This is narrower than "we invented disposition adapters" and more defensible.

### 11.6 Best concrete variant — confidence-gated differential sidecar

Given §11.5, Test 1 in §11.3 (the calibrated hedging head) is the cheapest diagnostic of the uncertainty-gating component, but the most complete realization of the full claim is:

**Confidence-gated differential sidecar.**
- Frozen base model (Gemma 4 E2B or Qwen3-7B).
- Small sidecar (~10–100M params) trained on `(base_output → disposition-corrected_output)` **deltas**, not full responses. Training data is the edit operation, not the replacement text.
- Per-token or per-span **gate** uses base signals: entropy, top-2 logit margin, optionally task/stage flags (e.g., "in final answer span" vs "in think block").
- Sidecar influence is **clipped and advisory**, not authoritative — the gate can veto the sidecar on high-confidence base spans, and the sidecar's edit magnitude is capped so it cannot rewrite base content wholesale.
- Evaluated against the Chef specialist verifier contract (gold checklist + dual criterion), not against a generic judge.

This is more novel than a plain Aligner clone (Aligner is uniform, not gated), more plausible than deep base-architecture surgery (no mechanistic claims about base circuits), and it is the cleanest test of the three-constraint combination. It sits between Test 1 (cheapest, narrowest) and Test 2 (broadest rewrite surface) in §11.3, and if Test 1 passes its falsifiable bar, this is the version to build out.

### 11.6b Results log — Test 0 (v0: correctness probe)

**Date:** 2026-04-09
**Question:** Does Gemma 4 E2B's per-token entropy / top-2 margin / final-hidden-state carry signal to distinguish correct from wrong responses on the existing Chef baseline? This is the prerequisite for any gated adapter: if base itself cannot be introspected to tell right from wrong, no gate can be trained.
**Gate:** AUC > 0.65 → proceed to v1. AUC ≤ 0.65 → frozen-base branch is dead.

**Setup**
- Data: `baseline_gemma4_v2_from_step16.json` — 100 Chef prompts, Gemma 4 E2B in bf16, judged by DeepSeek V3.2. Class balance: 58 correct, 42 wrong. Baseline assertion_asymmetry = −0.009, cells cr=53 / hr=5 / cw=38 / hw=4.
- Features: forward-pass each (prompt, response) pair; at each response-token position capture (entropy, top-2 logit margin, NLL); additionally capture the final hidden state (last layer, [1536-d]) at the mean and at the last response token.
- Classifier: L2 logistic regression, 5-fold stratified CV, standardized per fold.
- Script: `dd-v2/eval/uncert_probe_v0.py`. Results: `dd-v2/eval/results/uncert_probe_v0.json`.

**Results**

| Feature set | 5-fold AUC |
|---|---|
| Scalar 21-d (entropy/margin/NLL stats over response) | **0.566 ± 0.077** |
| Last-layer hidden state, mean over response tokens | **0.575 ± 0.109** |
| Last-layer hidden state, **final response token** | **0.683 ± 0.124** |
| Scalar + hidden-state mean | 0.600 ± 0.097 |

Folds (hs_last): 0.485, 0.596, 0.740, 0.812, 0.781.

**Verdict: GATE PASSES (0.683 > 0.65) on hs_last.** Branch stays alive. Proceed to v1.

**Interpretation**
1. **Scalar uncertainty features carry almost no signal (~0.57).** This is consistent with the Path B attribution result that Gemma's assertion_asymmetry is −0.009 at the verbal level: the model asserts equally confidently on right and wrong answers, so surface-level entropy and top-2 margin do not separate them. This alone kills a naive "high-entropy → hedge" gate.
2. **Hidden-state features carry modest signal (0.58–0.68), concentrated at the final response token.** Something in the residual stream at generation-end encodes correctness even when the output distribution does not reflect it. This is the architectural load-bearing fact for §11.6: the gate must read internal state, not output entropy.
3. **Effect size is modest.** AUC 0.68 with ±0.12 fold std on n=100 is a wide CI; the lower bound of a rough 95% CI is around 0.58. The signal is real but not overwhelming, and v1 needs ≥1k items to measure it cleanly.
4. **No large ceiling is claimed.** v0 only says the gate is *learnable in principle*. It says nothing yet about whether a learned gate can move the dual criterion (assertion_asym ≤ −0.15 while preserving coverage ≥ 0.43). That is v1's job.

**What this means for §11.6**
The confidence-gated differential sidecar design survives the first cheap test. Specifically:
- The "confidence gate" input should not be entropy alone — it should include the final-token hidden state (or a small pooled window around it).
- The novelty constraint "uncertainty-gated" is operationalized as "hidden-state-gated"; scalar uncertainty is too weak on Gemma 4 E2B's Chef distribution to carry the gate.
- The dual criterion has not been relaxed; v0 only clears the prerequisite.

Cost: ~$0.20 GPU on RunPod 4090 (model load + 100 forward passes + sklearn LR), zero API.

---

### 11.6c Results log — Test 1a (v1a: supervised gate ceiling, same-data)

**Date:** 2026-04-09
**Question:** Given the hs_last signal from v0, can a supervised gate with access to perfect labels reach the dual criterion `(assertion_asymmetry ≤ −0.15, coverage ≥ 0.43)` on the Gemma 4 E2B Chef distribution? If not, reward training cannot rescue it.
**Gate:** best balanced operating point (p_assert_correct ≥ 0.70) achieves `asym ≤ −0.15`.

**Setup — dtype alignment note**
Step 16's baseline was generated under Unsloth `FastModel.from_pretrained(..., load_in_4bit=True, full_finetuning=False)` with `FastModel.for_inference(model)` and greedy decoding. The correctness labels (`cr/hr/cw/hw`) therefore reflect the 4-bit quantized model's output, not a bf16 model. v0 was run in transformers bf16, which is not the same computation even though it uses the same weights. We reran v1a under the step16 load path exactly — Unsloth FastModel, `load_in_4bit=True`, `FastModel.for_inference`, multimodal chat template `[{"role":"user","content":[{"type":"text","text":p}]}]`, tokenization via `text_tok = tok.tokenizer` — so the hidden states we classify come from the same model state that produced the labels.

**Setup — method**
- Same 100 items from `baseline_gemma4_v2_from_step16.json`.
- Forward-pass each (prompt, response) pair. Extract three feature sets: scalar (21-d: entropy/margin/NLL stats over response tokens + length), `hs_mean` (mean last-layer hidden state over response tokens), `hs_last` (last-layer hidden state at the final response token).
- Classifiers: L2 LogReg (C=1.0), LogReg (C=0.1), MLPClassifier hidden=(32,). 5-fold stratified CV; for each fold, fit on train, predict on held-out test; concatenate OOF predictions. Standardized features.
- Gate sweep: across thresholds 0.05…0.95, compute `p_assert_wrong`, `p_assert_correct`, `asym`. Report both the unconstrained minimum and the minimum subject to `p_assert_correct ≥ 0.70` (the "balanced" operating point, so we don't trivially minimize asym by refusing to assert anything).
- Coverage is unchanged by definition — the gate is a hedge/no-hedge decision over otherwise identical content.
- Script: `dd-v2/eval/uncert_probe_v1a_4bit.py`. Results: `dd-v2/eval/results/uncert_probe_v1a_4bit.json`. A bf16 version (`uncert_probe_v1a.py`) was also run; numbers agree within noise, confirming the signal is not a dtype artifact.

**Results (4-bit, dtype-aligned)**

| Features | OOF AUC | Best asym | Balanced (p_assert_correct ≥ 0.70) |
|---|---|---|---|
| Scalar (21-d)        | 0.458 | −0.058 | −0.044 |
| Scalar + MLP(32)     | 0.534 | −0.196 | −0.100 |
| Hidden state mean    | 0.55  | −0.26  | −0.26  |
| hs_last + LogReg C=1 | 0.662 | −0.306 | **−0.306** |
| **hs_last + LogReg C=0.1** | **0.663** | **−0.341** | **−0.341** |
| hs_last + MLP(32)    | 0.634 | −0.268 | −0.183 |

**Baseline assertion_asymmetry = −0.009. Bar = −0.15. Best achieved = −0.341.**

**bf16 cross-check** (`uncert_probe_v1a.py`): hs_last + LogReg C=0.1 AUC=0.679, balanced asym=−0.296. Both dtypes agree the ceiling is well past the bar.

**Verdict: GATE PASSES by ~2× margin at the balanced operating point.**

**Interpretation**
1. **Scalar uncertainty features continue to fail.** 4-bit Gemma asserts with essentially uniform surface confidence on right and wrong answers; entropy/margin/NLL features give AUC 0.46–0.53. Any design that relies on output-level uncertainty alone is dead on arrival.
2. **Hidden-state features remain the only live signal**, concentrated at the final response token. LogReg beats MLP at n=100 — the signal is roughly linear in the residual stream and MLP overfits the small sample.
3. **Dtype alignment did not break the result.** Raw AUC shifted slightly (bf16 0.68 → 4-bit 0.66) but both sides of that shift are above the v0 gate and produce balanced gate asym ≤ −0.30. The bf16 result was not an artifact.
4. **The ceiling is comfortably above the bar.** The bar is −0.15; we hit −0.34. Even generous shrinkage to account for CV optimism still leaves headroom.

**Important caveat — this is not yet a real ceiling**
- n=100 with 5-fold CV means each held-out fold has 20 items. OOF predictions are honest (no leakage in the classifier), but the prompts come from the same 100-item distribution the gate saw in training on other folds; distributional overfit to the 100-prompt pool is possible.
- The gate has not been evaluated on held-out prompts drawn independently from the Chef distribution.
- To trust −0.34 as a real ceiling rather than a CV artifact, we need an independent test set — ideally ~1k fresh prompts with judge-computed labels. That is **v1-full**.

**Cost so far**
~$0.40 GPU total (v0 + v1a bf16 + v1a 4-bit), zero API. Next step (v1-full) is the first step with real API cost (~$5).

**What v1-full would do**
1. Generate ~1,000 new Chef prompts using the templated generator already used in step16 data prep.
2. Run 4-bit Gemma 4 E2B greedy on all 1k; capture hidden states at final response token.
3. Judge each response with DeepSeek V3.2 (same judge as step16) for `correct` and `verbal` labels.
4. Split train 800 / val 100 / test 100 with stratification on `correct`.
5. Train gate on 800; pick threshold on val; report asym on test.
6. Gate: test-set balanced asym ≤ −0.15.

If v1-full passes, v2 (GRPO) becomes the remaining question. If v1-full fails, the −0.34 here was CV-optimism and the frozen-base branch is much weaker than it looks at n=100.

---

### 11.6d Results log — Test 1-full (v1-full: honest fresh-prompt generalization)

**Date:** 2026-04-09
**Status:** **FALSIFIED on the factual_accuracy capability anchor.**
**Finding:** `FINDING_v1_full_v2_honest_fail.md`
**Scripts:** `dd-v2/eval/v1_full_v2.py`, `dd-v2/eval/v1_full_v2_batch_run.py`

**Setup.** 193 fresh French-Chef error-correction prompts generated by DeepSeek V3.2-exp, matching the step16 type1 shape. Gemma 4 E2B-it responses (Unsloth 4-bit, greedy, 512 tokens, batch=8 — dtype-aligned to step16). Per-claim gold-checklist scoring → `label = coverage ≥ 0.5`. Fresh positive rate 55/193 = 0.285. Step16 hs_last (n=100, pos=58) trains the gate. Fresh hs_last evaluates.

**Result.**

| Metric | Bar | v1a CV (n=100) | **v1-full fresh (n=193)** |
|---|---|---|---|
| AUC (factual correctness) | ≥ 0.63 | 0.663 / 0.683 | **0.516** |
| Balanced asym @ p_c ≥ 0.70 | ≤ −0.15 | −0.341 | **−0.043** |
| Gate | pass | pass | **fail** |

**Root cause (visible in the samples).** Gemma 4 E2B-it has a single output distribution on Chef error-correction prompts: verbose, structured, confident. Whether the model hits 3/5 canonical claims (label=1) or 1/5 (label=0), the final-token hidden state encodes the same stylistic closure, not the truth-value of the claims inside. Spot-check of 5 correct + 5 wrong confirms the response templates are interchangeable across classes. The v1a CV AUC of 0.68 was almost certainly picking up residual topic/length/category correlations within the 100 step16 items that do not transfer under independent sampling.

**Decision (per 11.6 decision matrix).** This is the "signal exists in CV but not on fresh → frozen-base sidecar foundation is too weak" row. T1-lite is moot at this surface (same gate, same numbers). The disposition probe sweep is being run only for completeness of the negative (expected to be flat across all 4 locked Chef axes — factual was the axis most likely to carry a clean signal and it failed). One cheap multi-layer pooling variant (mean of last 4 hidden layers) is being tried as a last probe-architecture sanity check before the branch is closed.

**Implication.** If the multi-layer variant is also flat on fresh, the §11.6 frozen-base branch is closed: the theory that Gemma 4 E2B's residual stream carries a correctness-correlated direction at the final response token is falsified on Chef, and the $55 GRPO budget is **not** authorized — a learned content-editing sidecar would be training a head to read a signal that demonstrably is not there. Main pivot (§7) remains intact: the contribution is the cross-arc negative, now with §11.6 closed underneath it rather than open.

---

### 11.6e Closure — sweep + multi-layer + spot-check

**Date:** 2026-04-09 (same session)
**Status:** **§11.6 CLOSED.** Every fallback inside the frozen-base sidecar theory now fails on fresh Chef prompts.
**Finding:** `FINDING_v1_full_v2_honest_fail.md`
**Closure report:** `REPORT_ddv2_11_6_closure.md`
**Scripts:** `dd-v2/eval/disposition_probe_sweep.py`, `dd-v2/eval/probe_multilayer.py`, `dd-v2/eval/eval_judge_rubric_v4.py`

**1. Disposition probe sweep (all 4 Chef axes + factual anchor).** Labels from v4 5-axis judge (response-blind 2-pass, binarized at score ≥4). LogReg(C=0.1) on standardized step16 hs_last, 5-fold CV on n=100, train→fresh transfer on n=193.

| Axis | Disposition | CV AUC (bar 0.65) | Transfer AUC (bar 0.60) | Pass |
|---|---|---|---|---|
| factual_accuracy | capability anchor | 0.533 | **0.480** | ❌ |
| hedging_appropriateness | Humble | 0.599 | **0.515** | ❌ |
| pedagogical_framing | Deliberate | 0.481 | **0.439** | ❌ |
| self_verification | Adversarial-self | 0.594 | **0.518** | ❌ |
| completeness | Persistent | 0.510 | **0.442** | ❌ |

0/4 disposition axes pass. Three axes (factual / pedagogical / completeness) have transfer AUC **below chance**, confirming the step16 CV was fitting noise. Cross-axis correlations all |r| < 0.8 (clean separability, but meaningless when no axis has signal).

**2. Multi-layer probe variant (mean of last 4 hidden layers).** Same LogReg recipe, different feature.

| Config | Fresh AUC (bar 0.60) | Balanced asym @ p_c≥0.70 (bar −0.15) | Pass |
|---|---|---|---|
| last-4 mean, C=1.0 | 0.553 | −0.123 | ❌ |
| last-4 mean, C=0.1 | **0.557** | **−0.130** | ❌ (closest) |
| hs_last (v1-full reference) | 0.516 | −0.043 | ❌ |

Mean-pooling the last 4 layers recovers modest signal — asym improves from −0.04 to −0.13 — but still does not clear the −0.15 bar, and AUC remains far below 0.63. **Closes the "wrong layer" alibi.**

**3. Behavioral spot-check (5 low-score samples per axis).** v4 judge labels are defensible. Key patterns in the fresh responses:
- **Adversarial-self is the weakest disposition:** 78/192 (41%) score ≤2. Gemma explains and enumerates but does not cross-check ("if X then Y", "this assumes…").
- **Pedagogical is near-ceiling:** only 12/192 (6%) score low. Structurally pedagogical by default.
- **Hedging failures are "clarifying question on CERTAIN prompt":** Gemma says "I need a little more information" on canonically answerable questions.
- **Catastrophic factual misses do occur:** idx=38 "blanched green beans olive drab" → model answers about **watering houseplants**. Judge caught it cleanly.

Behavior varies; the judge reads it; the probe cannot. The signal distinguishing responses lives in the token distribution of the output, not in the final-position (or last-4-mean) hidden state.

**Closure.** §11.6 is falsified along every dimension it was specified under:
1. `hs_last` probe on fresh Chef factual: AUC 0.52, asym −0.04. ❌
2. Per-axis `hs_last` probe on 4 Chef dispositions: 0/4 pass, several below chance. ❌
3. Multi-layer (last-4 mean) pooling: fresh AUC 0.56, asym −0.13. ❌
4. T1-lite behavioral test: moot (same gate surface).

The §55 GRPO budget is **not** authorized: a learned content-editing sidecar would be training a head to read a signal that provably is not there on this base / this domain / this scale. The main pivot (§7) remains intact; §11.6 is now **closed** rather than **open**, which strengthens the cross-arc negative by removing the standing "frozen-base alternative" footnote.

**What this does NOT falsify:** execution-grounded process training on a verifiable domain (code + HumanEval/MBPP oracle) for a functional trait (persistence or adversarial-self), which reads the signal from *outcomes of behavior* rather than from internal probes. That is the remaining candidate for a positive DD v2 contribution and is under discussion separately — it is not an extension of §11.6.

---

### 11.6f Cross-model replication on SmolLM2-1.7B-Instruct (2026-04-09)

**Concern:** §11.6 was run entirely on `unsloth/gemma-4-E2B-it` in 4-bit. A reader could object that the failure is a Gemma-instruct-tune artifact — this model happens to have an unusually uniform output distribution on Chef error-correction prompts, so `hs_last` is too templated to carry truth-value, but *other* sub-2B instruct models might carry readable signal.

**Test:** re-run the factual anchor (the axis most likely to carry clean `hs_last` signal) on a second frozen base: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, bf16, HF transformers. Same step16 prompts, same fresh 193, same gold checklists, same LogReg probe, same gate.

**Scripts:** `smollm_sweep_run.py` (feature extraction + fresh generation), `smollm_factual_score.py` (gold scoring + probe fit).

**Results:**

```
step16:  n=100  pos=56
fresh:   n=193  pos=9          ← SmolLM capability cliff (vs Gemma pos=55)
C=1.0:   AUC=0.519   asym=-0.161 @ thr=0.06   (degenerate)
C=0.1:   AUC=0.505   asym=-0.188 @ thr=0.16   (degenerate)
gate_pass = False
```

The sub-threshold asym numbers nominally clear −0.15 but are **degenerate class-imbalance artifacts** with only 9/193 positives — at thr=0.06 the probe predicts "assert" on nearly everything, trivially inflating `p_correct` near 1.0. AUC at chance (0.52) confirms there is no real direction.

**Two distinct failure modes, one conclusion:**

| Failure route     | Gemma 4 E2B                         | SmolLM2-1.7B                           |
|-------------------|-------------------------------------|----------------------------------------|
| Mechanism         | Single output distribution          | Capability cliff (step16 → fresh)      |
| fresh pos rate    | 0.285                               | 0.047                                  |
| fresh AUC         | 0.516                               | 0.519                                  |
| Gate              | ❌                                  | ❌                                     |

**Cross-model conclusion (added to closure):** at sub-2B instruct scale on judge-based domains, the frozen-base last-layer residual stream at the final response token does **not** carry a content-editable correctness direction that a standard linear probe can read. The §11.6 closure is not specific to Gemma's instruct-tune — it is a property of this class of model on this class of domain. The SmolLM disposition-axis sweep was not run: factual anchor already failed, and `fresh_pos=9` has no statistical power for a 4-way split.

**Two models, two failure routes, same verdict. §11.6 closure reinforced.**

---

### 11.7 Relation to the main pivot

The §7 pivot ("accept falsification at this scale, write the cross-arc negative as the main contribution") stands unchanged. This branch is an **open thread**, not a reopening. It is recorded here so that:

1. Future work has an explicit, scoped, motivated next experiment rather than a blanket "DD is closed."
2. The negative in §§ 2–10 is not mistaken for a claim that *any* form of disposition shaping is impossible — it is a claim about weight-modifying operators specifically.
3. If Test 1 is ever run, its result cleanly updates the negative in one of two directions: a positive narrows the negative to "imitation-into-weights doesn't work"; a negative strengthens it to "even frozen-base reward-trained adapters don't work at this scale."

Either outcome is more informative than leaving the branch unstated.
