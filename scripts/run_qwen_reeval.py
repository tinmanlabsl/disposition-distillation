#!/usr/bin/env python3
"""
Qwen3.5-0.8B HumanEval re-eval with official recommended sampling parameters.
Previous run: 24.4% with greedy decoding (do_sample=False).
Qwen explicitly warns: "DO NOT use greedy decoding."

Recommended for non-thinking mode text tasks:
  temperature=1.0, top_p=1.00, top_k=20, presence_penalty=2.0

Recommended for thinking mode coding tasks:
  temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0

We run BOTH modes and compare.
"""
import json, time, re, torch, subprocess, gc, signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

CACHE = "/workspace/persistent/eval/humaneval_problems.json"
RESULTS_DIR = "/workspace/persistent/eval"


def gpu_check(label):
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"  GPU[{label}]: {r.stdout.strip()}", flush=True)


def extract_code(resp):
    text = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def check_solution(prob, completion):
    code = prob["prompt"] + completion + "\n" + prob["test"] + "\ncheck(" + prob["entry_point"] + ")"
    try:
        exec(code, {})
        return True
    except Exception:
        return False


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}", flush=True)


def run_eval(mode, tok, model, problems, gen_kwargs):
    """Run HumanEval with given generation kwargs."""
    print(f"\n  --- Mode: {mode} ---", flush=True)
    results = []
    passed = 0
    start = time.time()

    for i, p in enumerate(problems):
        prompt_text = (
            "Complete the following Python function. "
            "Output ONLY the function body code, no explanation.\n\n" + p["prompt"]
        )
        messages = [{"role": "user", "content": prompt_text}]

        enable_thinking = mode == "thinking"
        try:
            input_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        except TypeError:
            input_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        ids = tok([input_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, pad_token_id=tok.eos_token_id, **gen_kwargs)
        resp = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        comp = extract_code(resp)
        ok = check_solution(p, comp)
        if ok:
            passed += 1

        results.append({
            "task_id": p["task_id"], "passed": ok,
            "has_think": "<think>" in resp, "resp_len": len(resp),
            "completion": comp[:500], "raw": resp[:500],
        })

        if (i + 1) % 20 == 0:
            el = time.time() - start
            rate = (i + 1) / el * 60 if el > 0 else 0
            print(f"  [{i+1}/164] {passed}/{i+1} ({passed/(i+1)*100:.1f}%) | {rate:.0f} prob/min", flush=True)

    el = time.time() - start
    score = passed / len(problems) * 100
    think_ct = sum(1 for r in results if r["has_think"])
    print(f"\n  {mode} RESULT: Pass@1 = {passed}/{len(problems)} = {score:.1f}%  Think: {think_ct}  Time: {el:.0f}s", flush=True)

    return {
        "mode": mode, "score": score, "passed": passed,
        "total": len(problems), "think_blocks": think_ct,
        "elapsed": el, "results": results,
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading Qwen3.5-0.8B...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        "/workspace/persistent/models/qwen3.5-0.8b", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "/workspace/persistent/models/qwen3.5-0.8b",
        dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    gpu_check("after_load")

    with open(CACHE) as f:
        problems = json.load(f)

    all_results = {}

    # Mode 1: Non-thinking with recommended sampling (fast, fair comparison)
    # temperature=0.6, top_p=0.95, top_k=20 (coding task recommended)
    print("\n" + "=" * 60)
    print("Qwen3.5-0.8B HumanEval: NON-THINKING + SAMPLING (coding recommended)")
    print("=" * 60, flush=True)
    all_results["non_thinking_sampled"] = run_eval("non_thinking", tok, model, problems, {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    })
    save_json(all_results, f"{RESULTS_DIR}/qwen35_reeval_results.json")

    # Mode 2: Non-thinking with repetition penalty (addresses echo/repetition)
    print("\n" + "=" * 60)
    print("Qwen3.5-0.8B HumanEval: NON-THINKING + REP PENALTY")
    print("=" * 60, flush=True)
    all_results["non_thinking_rep_penalty"] = run_eval("non_thinking", tok, model, problems, {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "repetition_penalty": 1.5,
    })
    save_json(all_results, f"{RESULTS_DIR}/qwen35_reeval_results.json")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Greedy (previous):  24.4% (40/164)")
    for mode, r in all_results.items():
        print(f"  {mode}: {r['score']:.1f}% ({r['passed']}/{r['total']}) [think={r['think_blocks']}]")
    print(flush=True)
