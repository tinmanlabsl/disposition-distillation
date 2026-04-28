# Disposition Distillation at Sub-Billion Scale: A Three-Arc Negative Result

Artifact repository for the paper *"Disposition Distillation at Sub-Billion Scale: A Three-Arc Negative Result"* — [arXiv:2604.11867](https://arxiv.org/abs/2604.11867).

We set out to train behavioral dispositions — self-verification, uncertainty acknowledgment, feedback integration — into sub-2B language models. Across three independent operator classes, we find no operator that moves judge-measured disposition without simultaneously damaging content quality or collapsing into stylistic mimicry:

1. **Arc 1 — Imitation-into-weights** via SFT + DPO LoRA (Qwen3-0.6B, Qwen3-1.7B). The originally-reported +15.3pt HumanEval and +33.9pt MCAS deltas did not survive honest re-evaluation; corrected deltas are −8.0pt and −0.32pt respectively.
2. **Arc 2 — Inference-time attention-head tempering** on `o_proj` rows (Gemma 4 E2B, Qwen3.5-0.8B, two dispositions, five variants). Disposition either does not move, moves the wrong direction, or moves only at the cost of content damage.
3. **Arc 3 — Frozen-base `h_last` confidence-gated sidecar** (Gemma 4 E2B → SmolLM2-1.7B-Instruct cross-model replication). A within-distribution CV ceiling of AUC 0.683 collapses to AUC 0.516–0.557 on fresh prompts from the same distribution, judge, and model — and to AUC 0.519 on a second base, with a distinct mechanism (capability cliff vs uniform output distribution).

The contribution is the cross-arc negative with mechanism: at this scale, the residual stream at the final response token encodes stylistic closure, not content truth-value, and that gap explains all three failures.

## Repository layout

```
paper/
  dd-paper-arxiv-v3.pdf         Final paper (v3)
  figs/                         Figure renderer + 4 PDFs
findings/                       28 markdown research notes — the substrate
                                of the paper, including approach docs,
                                falsification reports, and the closure report
eval/
  scripts/                      Probes, judges, attribution, sweeps
  results/                      JSON / NPZ artifacts the paper cites
training/                       SFT, DPO, ablation training scripts (Arc 1)
verification/                   Format validator + chef domain verifier
scripts/                        Data-generation pipeline (4-stage MIT teacher
                                stack) and baseline runners
phase0_audit/                   Arc 1 corrected re-evaluation (the audit
                                that flipped +15.3 → −8.0 on HumanEval)
```

## Reproducibility notes

- All training and evaluation was run on RunPod RTX 4090 pods with a Network Volume mounted at `/workspace/persistent/`. Hard-coded paths in `eval/scripts/*.py` and `scripts/*.py` reflect that mount; substitute your own paths.
- Teacher models are accessed via paid APIs (Z.ai for GLM-5, Moonshot for Kimi K2.5, MiniMax, Together / Novita for DeepSeek). Set credentials in your environment; the scripts read from `os.environ`.
- The 100-prompt Step 16 Chef eval set and the 193-prompt fresh held-out set are not redistributed in this repo. The judge rubric (`eval/scripts/eval_judge_rubric_v4.py`) and verifier (`verification/chef_verifier.py`) are sufficient to reproduce evaluation against your own equivalent prompt set.
- Figures are reproducible from the JSON / NPZ artifacts in `eval/results/` via `paper/figs/make_figures.py`.

## Teacher stack

100% MIT-licensed teacher models (Kimi K2.5, GLM-5, MiniMax M2.7, Qwen3.6+ / DeepSeek V3.2 for fresh prompt synthesis). Claude was used for some early eval-only judging; the paper's primary judge is a 5-axis rubric run on DeepSeek V3.2. No Claude or OpenAI outputs entered training data.

## How to cite

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
