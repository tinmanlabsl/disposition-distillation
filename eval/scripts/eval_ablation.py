"""
eval_ablation.py -- Run full ablation comparison across 4 DD v2 conditions.

Conditions:
  a) Baseline: no training (base model)
  b) Gold-ST:  gold-only, single-teacher SFT (Type 1, Gemma 4 only)
  c) Gold-MT:  gold-only, multi-teacher SFT (Type 1, full 4-stage pipeline)
  d) Full DDv2: all types + DPO

Usage:
    python eval_ablation.py \
        --baseline Qwen/Qwen3-0.6B \
        --gold-single ./checkpoints/gold-single \
        --gold-multi  ./checkpoints/gold-multi \
        --full-ddv2   ./checkpoints/full-ddv2 \
        --eval-data ../data/eval/chef_eval.jsonl
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing ChefVerifier from the verification directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR / "verification"))
from chef_verifier import ChefVerifier


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Load a HuggingFace model + tokenizer. Handles Qwen3.5, SmolLM3, etc."""
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate a single response with greedy decoding."""
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Think-block check
# ---------------------------------------------------------------------------

def check_think_blocks(response: str) -> dict:
    open_tags = len(re.findall(r"<think>", response))
    close_tags = len(re.findall(r"</think>", response))
    return {
        "open_tags": open_tags,
        "close_tags": close_tags,
        "unclosed": max(0, open_tags - close_tags),
        "has_think": open_tags > 0,
    }


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    eval_data: list[dict],
    verifier: ChefVerifier,
    label: str = "Model",
) -> dict:
    """Run full ChefVerifier evaluation on one model. Returns metrics dict."""
    technique_correct, technique_incorrect = 0, 0
    regional_correct, regional_incorrect = 0, 0
    ingredient_correct, ingredient_incorrect = 0, 0
    invented_count, known_count = 0, 0
    scores = []
    think_block_total = 0
    unclosed_think_total = 0
    responses_with_think = 0

    per_prompt_results = []

    print(f"\n  [{label}] evaluating {len(eval_data)} prompts ...")
    t0 = time.time()

    for i, item in enumerate(eval_data):
        prompt = item.get("prompt", item.get("text", ""))
        if not prompt:
            continue

        response = generate_response(model, tokenizer, prompt)
        verification = verifier.verify_response(response, prompt)
        score = verifier.score_response(response, prompt)
        scores.append(score)

        checks = verification["checks"]

        t_check = checks["technique_accuracy"]
        technique_correct += len(t_check["correct"])
        technique_incorrect += len(t_check["incorrect"])

        r_check = checks["regional_attribution"]
        regional_correct += len(r_check["correct"])
        regional_incorrect += len(r_check["incorrect"])

        i_check = checks["ingredient_correctness"]
        ingredient_correct += len(i_check["correct"])
        ingredient_incorrect += len(i_check["incorrect"])

        inv_check = checks["invented_references"]
        invented_count += len(inv_check["potentially_invented"])
        known_count += len(inv_check["known"])

        tb = check_think_blocks(response)
        if tb["has_think"]:
            responses_with_think += 1
        think_block_total += tb["open_tags"]
        unclosed_think_total += tb["unclosed"]

        per_prompt_results.append({
            "prompt": prompt[:200],
            "score": score,
            "verification_passed": verification["passed"],
            "think_blocks": tb,
        })

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(eval_data)} done ...")

    elapsed = time.time() - t0

    def safe_acc(c, inc):
        total = c + inc
        return c / total if total > 0 else float("nan")

    def safe_rate(num, denom):
        return num / denom if denom > 0 else float("nan")

    metrics = {
        "label": label,
        "num_prompts": len(eval_data),
        "technique_accuracy": safe_acc(technique_correct, technique_incorrect),
        "technique_correct": technique_correct,
        "technique_incorrect": technique_incorrect,
        "regional_accuracy": safe_acc(regional_correct, regional_incorrect),
        "regional_correct": regional_correct,
        "regional_incorrect": regional_incorrect,
        "ingredient_accuracy": safe_acc(ingredient_correct, ingredient_incorrect),
        "ingredient_correct": ingredient_correct,
        "ingredient_incorrect": ingredient_incorrect,
        "invented_reference_rate": safe_rate(invented_count, invented_count + known_count),
        "invented_count": invented_count,
        "known_count": known_count,
        "avg_verifier_score": sum(scores) / len(scores) if scores else 0.0,
        "think_blocks_total": think_block_total,
        "unclosed_think_blocks": unclosed_think_total,
        "responses_with_think": responses_with_think,
        "elapsed_seconds": round(elapsed, 1),
    }

    return {
        "metrics": metrics,
        "per_prompt_results": per_prompt_results,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def fmt_pct(val: float) -> str:
    if val != val:  # NaN
        return "N/A"
    return f"{val * 100:.1f}%"


def print_ablation_table(all_metrics: list[dict]):
    """Print the ablation comparison table."""
    header = (
        f"{'Condition':<14} | {'Tech Acc':>9} | {'Reg Acc':>9} | {'Ingr Acc':>9} "
        f"| {'Inv Rate':>9} | {'Avg Score':>9} | {'Think Blk':>9}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for m in all_metrics:
        row = (
            f"{m['label']:<14} | {fmt_pct(m['technique_accuracy']):>9} "
            f"| {fmt_pct(m['regional_accuracy']):>9} "
            f"| {fmt_pct(m['ingredient_accuracy']):>9} "
            f"| {fmt_pct(m['invented_reference_rate']):>9} "
            f"| {m['avg_verifier_score']:>9.3f} "
            f"| {m['think_blocks_total']:>9}"
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS = [
    ("baseline",     "Baseline",  "No training (base model)"),
    ("gold_single",  "Gold-ST",   "Gold-only, single-teacher SFT (Type 1, Gemma 4)"),
    ("gold_multi",   "Gold-MT",   "Gold-only, multi-teacher SFT (Type 1, full pipeline)"),
    ("full_ddv2",    "Full DDv2", "All types + DPO"),
]


def main():
    parser = argparse.ArgumentParser(
        description="DD v2 ablation: compare 4 training conditions on French Chef eval"
    )
    parser.add_argument("--baseline", required=True,
                        help="Path to untrained baseline model")
    parser.add_argument("--gold-single", required=True,
                        help="Path to gold-only single-teacher SFT model")
    parser.add_argument("--gold-multi", required=True,
                        help="Path to gold-only multi-teacher SFT model")
    parser.add_argument("--full-ddv2", required=True,
                        help="Path to full DD v2 model (all types + DPO)")
    parser.add_argument("--eval-data",
                        default=str(_PROJECT_DIR / "data" / "eval" / "chef_eval.jsonl"),
                        help="Path to eval JSONL file")
    parser.add_argument("--ref-db",
                        default=str(_PROJECT_DIR / "data" / "french_cuisine_reference.json"),
                        help="Path to reference DB JSON")
    parser.add_argument("--output",
                        default=str(_PROJECT_DIR / "results" / "chef_ablation.json"),
                        help="Path to save ablation results JSON")

    args = parser.parse_args()

    # Map condition keys to CLI paths
    model_paths = {
        "baseline":    args.baseline,
        "gold_single": args.gold_single,
        "gold_multi":  args.gold_multi,
        "full_ddv2":   args.full_ddv2,
    }

    # Load eval data
    eval_data = []
    with open(args.eval_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))
    print(f"Loaded {len(eval_data)} eval prompts from {args.eval_data}")

    # Load verifier
    verifier = ChefVerifier(db_path=args.ref_db)

    # Evaluate each condition sequentially (free GPU between models)
    all_results = []
    all_metrics = []

    for cond_key, cond_label, cond_desc in ABLATION_CONDITIONS:
        path = model_paths[cond_key]
        print(f"\n{'='*60}")
        print(f"Condition: {cond_label} -- {cond_desc}")
        print(f"Model:     {path}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(path)
        result = evaluate_model(model, tokenizer, eval_data, verifier, label=cond_label)

        all_results.append(result)
        all_metrics.append(result["metrics"])

        # Free GPU memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print ablation table
    print_ablation_table(all_metrics)

    # Think-block summary
    print("\nThink-block summary:")
    for m in all_metrics:
        flag = " <<< WARNING" if m["unclosed_think_blocks"] > 0 else ""
        print(f"  {m['label']:<14}: {m['think_blocks_total']} total, "
              f"{m['unclosed_think_blocks']} unclosed{flag}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "eval_data_path": args.eval_data,
        "ref_db_path": args.ref_db,
        "num_prompts": len(eval_data),
        "conditions": [
            {
                "key": cond_key,
                "label": cond_label,
                "description": cond_desc,
                "model_path": model_paths[cond_key],
            }
            for cond_key, cond_label, cond_desc in ABLATION_CONDITIONS
        ],
        "metrics": all_metrics,
        "per_condition_details": [
            {
                "label": r["metrics"]["label"],
                "per_prompt_results": r["per_prompt_results"],
            }
            for r in all_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nAblation results saved to {output_path}")


if __name__ == "__main__":
    main()
