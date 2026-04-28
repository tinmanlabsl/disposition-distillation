"""
eval_chef.py -- Evaluate a trained French Chef specialist model on
domain-specific metrics using ChefVerifier.

Usage:
    python eval_chef.py --model ./checkpoints/chef-ddv2 --eval-data ../data/eval/chef_eval.jsonl
    python eval_chef.py --model ./checkpoints/chef-ddv2 --baseline Qwen/Qwen3-0.6B --num-prompts 50
"""

import argparse
import json
import os
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
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate a single response with greedy decoding."""
    messages = [{"role": "user", "content": prompt}]
    # Use chat template if available, otherwise raw prompt
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
    # Decode only the generated portion
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Think-block sanity check
# ---------------------------------------------------------------------------

def check_think_blocks(response: str) -> dict:
    """Scan a response for <think>/<think> tags and unclosed blocks."""
    open_tags = len(re.findall(r"<think>", response))
    close_tags = len(re.findall(r"</think>", response))
    unclosed = open_tags - close_tags
    return {
        "open_tags": open_tags,
        "close_tags": close_tags,
        "unclosed": max(0, unclosed),
        "has_think": open_tags > 0,
    }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    eval_data: list[dict],
    verifier: ChefVerifier,
    num_prompts: int | None = None,
    label: str = "Model",
) -> dict:
    """Run evaluation on a model and return metrics + sample responses."""
    if num_prompts is not None:
        eval_data = eval_data[:num_prompts]

    # Accumulators
    technique_correct, technique_incorrect = 0, 0
    regional_correct, regional_incorrect = 0, 0
    ingredient_correct, ingredient_incorrect = 0, 0
    invented_count, known_count = 0, 0
    scores = []
    think_block_total = 0
    unclosed_think_total = 0
    responses_with_think = 0

    all_results = []
    sample_responses = []

    print(f"\nEvaluating [{label}] on {len(eval_data)} prompts ...")
    t0 = time.time()

    for i, item in enumerate(eval_data):
        prompt = item.get("prompt", item.get("text", ""))
        if not prompt:
            continue

        response = generate_response(model, tokenizer, prompt)

        # Verification
        verification = verifier.verify_response(response, prompt)
        score = verifier.score_response(response, prompt)
        scores.append(score)

        checks = verification["checks"]

        # Technique accuracy
        t_check = checks["technique_accuracy"]
        technique_correct += len(t_check["correct"])
        technique_incorrect += len(t_check["incorrect"])

        # Regional attribution
        r_check = checks["regional_attribution"]
        regional_correct += len(r_check["correct"])
        regional_incorrect += len(r_check["incorrect"])

        # Ingredient correctness
        i_check = checks["ingredient_correctness"]
        ingredient_correct += len(i_check["correct"])
        ingredient_incorrect += len(i_check["incorrect"])

        # Invented references
        inv_check = checks["invented_references"]
        invented_count += len(inv_check["potentially_invented"])
        known_count += len(inv_check["known"])

        # Think-block check
        tb = check_think_blocks(response)
        if tb["has_think"]:
            responses_with_think += 1
        think_block_total += tb["open_tags"]
        unclosed_think_total += tb["unclosed"]

        # Collect sample responses (first 10)
        if len(sample_responses) < 10:
            sample_responses.append({
                "prompt": prompt[:200],
                "response": response[:500],
                "score": score,
                "think_blocks": tb,
            })

        all_results.append({
            "prompt": prompt[:200],
            "score": score,
            "verification": verification,
            "think_blocks": tb,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(eval_data)} done ...")

    elapsed = time.time() - t0

    # Compute metrics
    def safe_acc(correct, incorrect):
        total = correct + incorrect
        return correct / total if total > 0 else float("nan")

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
        "all_results": all_results,
        "sample_responses": sample_responses,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def fmt_pct(val: float) -> str:
    """Format a float as percentage string, handling NaN."""
    if val != val:  # NaN check
        return "N/A"
    return f"{val * 100:.1f}%"


def print_results_table(results: list[dict]):
    """Print a formatted comparison table."""
    header = (
        f"{'Condition':<14} | {'Tech Acc':>9} | {'Reg Acc':>9} | {'Ingr Acc':>9} "
        f"| {'Inv Rate':>9} | {'Avg Score':>9} | {'Think Blk':>9}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        m = r["metrics"]
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


def print_samples(samples: list[dict], label: str):
    """Print sample responses for manual inspection."""
    print(f"\n{'='*60}")
    print(f" Sample Responses: {label}")
    print(f"{'='*60}")
    for i, s in enumerate(samples):
        print(f"\n--- Sample {i+1} (score: {s['score']:.3f}) ---")
        print(f"Prompt:   {s['prompt']}")
        print(f"Response: {s['response']}")
        if s["think_blocks"]["has_think"]:
            print(f"  [THINK BLOCKS: {s['think_blocks']['open_tags']} open, "
                  f"{s['think_blocks']['unclosed']} unclosed]")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a French Chef specialist model with ChefVerifier"
    )
    parser.add_argument("--model", required=True, help="Path to trained model (HF format)")
    parser.add_argument("--baseline", default=None, help="Path to untrained baseline model for comparison")
    parser.add_argument("--eval-data", default=str(_PROJECT_DIR / "data" / "eval" / "chef_eval.jsonl"),
                        help="Path to eval JSONL file")
    parser.add_argument("--ref-db", default=str(_PROJECT_DIR / "data" / "french_cuisine_reference.json"),
                        help="Path to reference DB JSON")
    parser.add_argument("--output", default=str(_PROJECT_DIR / "results" / "chef_results.json"),
                        help="Path to save results JSON")
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Limit eval to N prompts (for quick testing)")

    args = parser.parse_args()

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

    # Evaluate trained model
    model, tokenizer = load_model_and_tokenizer(args.model)
    trained_result = evaluate_model(
        model, tokenizer, eval_data, verifier,
        num_prompts=args.num_prompts, label="Trained",
    )
    # Free GPU memory before loading baseline
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_results = [trained_result]

    # Evaluate baseline if provided
    if args.baseline:
        bl_model, bl_tokenizer = load_model_and_tokenizer(args.baseline)
        baseline_result = evaluate_model(
            bl_model, bl_tokenizer, eval_data, verifier,
            num_prompts=args.num_prompts, label="Baseline",
        )
        del bl_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        all_results.insert(0, baseline_result)

    # Print results table
    print_results_table(all_results)

    # Print sample responses from trained model
    print_samples(trained_result["sample_responses"], "Trained Model")

    # Think-block summary
    m = trained_result["metrics"]
    print(f"Think-block sanity check:")
    print(f"  Responses with <think> tags: {m['responses_with_think']}/{m['num_prompts']}")
    print(f"  Total think blocks: {m['think_blocks_total']}")
    print(f"  Unclosed think blocks: {m['unclosed_think_blocks']}")
    if m["unclosed_think_blocks"] > 0:
        print(f"  WARNING: {m['unclosed_think_blocks']} unclosed think blocks detected!")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "eval_data_path": args.eval_data,
        "ref_db_path": args.ref_db,
        "num_prompts": args.num_prompts or len(eval_data),
        "results": [
            {
                "metrics": r["metrics"],
                "sample_responses": r["sample_responses"],
            }
            for r in all_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
