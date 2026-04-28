"""
HumanEval baseline + LoRA sanity check for 3 candidate models.
Designed for 5090 (32GB) - loads models one at a time.
"""
import json, os, sys, time, re, traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "qwen3.5-0.8b": "/workspace/persistent/models/qwen3.5-0.8b",
    "smollm3-3b": "/workspace/persistent/models/smollm3-3b",
    "gemma-4-e2b": "/workspace/persistent/models/gemma-4-e2b",
    "gemma-3-1b-it": "/workspace/persistent/models/gemma-3-1b-it",
}

def load_humaneval():
    cache = "/workspace/persistent/eval/humaneval_problems.json"
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    from datasets import load_dataset
    return list(load_dataset("openai/openai_humaneval", split="test"))

def extract_code(response):
    text = response
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if code_match:
        return code_match.group(1)
    return text

def check_solution(problem, completion):
    full_code = problem["prompt"] + completion + "\n" + problem["test"] + "\ncheck(" + problem["entry_point"] + ")"
    try:
        exec_globals = {}
        exec(full_code, exec_globals)
        return True
    except Exception:
        return False

def run_humaneval(model_name, model_path):
    print(f"\n{'='*60}")
    print(f"HumanEval: {model_name}")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, attn_implementation="sdpa"
    )
    model.eval()

    problems = load_humaneval()
    results = []
    passed = 0
    start = time.time()

    for i, problem in enumerate(problems):
        prompt_text = "Complete the following Python function. Output ONLY the function body code. No explanation.\n\n" + problem["prompt"]

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            try:
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                try:
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    input_text = prompt_text
        else:
            input_text = prompt_text

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_new_tokens=1024, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        completion = extract_code(response)
        correct = check_solution(problem, completion)
        if correct:
            passed += 1

        has_think = "<think>" in response

        results.append({
            "task_id": problem["task_id"],
            "passed": correct,
            "has_think_block": has_think,
            "response_len": len(response),
            "raw_response": response[:300],
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed * 60
            print(f"  [{i+1}/164] {passed}/{i+1} ({passed/(i+1)*100:.1f}%) | {rate:.0f} prob/min", flush=True)

    elapsed = time.time() - start
    score = passed / len(problems) * 100
    think_count = sum(1 for r in results if r["has_think_block"])

    print(f"\nRESULT: {model_name}")
    print(f"  Pass@1: {passed}/{len(problems)} = {score:.1f}%")
    print(f"  Think blocks: {think_count}/{len(problems)}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "score": score,
        "passed": passed,
        "total": len(problems),
        "think_blocks": think_count,
        "elapsed_seconds": elapsed,
        "results": results,
    }

def run_lora_sanity(model_name, model_path):
    print(f"\n{'='*60}")
    print(f"LoRA Sanity: {model_name}")
    print(f"{'='*60}", flush=True)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        os.system("pip install -q peft")
        from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, attn_implementation="sdpa"
    )

    # Always use explicit attention projection targets
    # Auto-detection can hit non-standard modules (e.g. Gemma4ClippableLinear)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    print(f"  Target modules: {target_modules}", flush=True)

    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=target_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 50 dummy training examples
    dummy_pairs = [
        ("What is a bechamel sauce?", "Bechamel is a white sauce made from butter, flour, and milk."),
        ("How do you make a roux?", "A roux is made by cooking equal parts flour and fat together."),
        ("What region is bouillabaisse from?", "Bouillabaisse is a traditional fish stew from Marseille, Provence."),
        ("Explain the julienne cut.", "Julienne is a knife cut producing thin strips, approximately 3mm x 3mm x 50mm."),
        ("What temperature for caramel?", "Caramel forms at approximately 160-177 degrees Celsius."),
        ("What are the five mother sauces?", "The five French mother sauces are bechamel, veloute, espagnole, hollandaise, and tomato."),
        ("What is a mirepoix?", "Mirepoix is a combination of diced onions, carrots, and celery, typically in a 2:1:1 ratio."),
        ("How do you deglaze a pan?", "Add liquid to a hot pan after searing to dissolve the fond (browned bits) from the bottom."),
        ("What is confit?", "Confit is a preservation method where meat is slowly cooked in its own fat at low temperature."),
        ("What region is choucroute from?", "Choucroute garnie is a traditional dish from Alsace, featuring sauerkraut with sausages and pork."),
    ] * 5  # 50 total

    train_texts = []
    for p, r in dummy_pairs:
        msgs = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        except TypeError:
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
            except Exception:
                text = "User: " + p + "\nAssistant: " + r
        train_texts.append(text)

    encodings = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=2e-4)
    model.train()

    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    total_loss = 0
    for batch_start in range(0, len(train_texts), 10):
        batch_ids = input_ids[batch_start:batch_start+10]
        batch_mask = attention_mask[batch_start:batch_start+10]
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        print(f"  Batch {batch_start//10 + 1}/5, loss={loss.item():.4f}", flush=True)

    # Test: generate 10 responses, check for format corruption
    model.eval()
    test_prompts = [
        "What are the five French mother sauces?",
        "How do you make hollandaise?",
        "What is the difference between saute and braise?",
        "What region is cassoulet from?",
        "What temperature should oil be for deep frying?",
        "Explain the technique of deglazing.",
        "What is a bouquet garni?",
        "How do you clarify butter?",
        "What knife cut is brunoise?",
        "Name three dishes from Burgundy.",
    ]

    issues = []
    think_blocks = 0
    garbage_outputs = 0

    for prompt in test_prompts:
        msgs = [{"role": "user", "content": prompt}]
        try:
            input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            try:
                input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = "User: " + prompt + "\nAssistant:"

        ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)

        resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        has_think = "<think>" in resp or "</think>" in resp
        unclosed_think = "<think>" in resp and "</think>" not in resp
        is_coherent = len(resp.strip()) > 10 and any(c.isalpha() for c in resp)

        if has_think:
            think_blocks += 1
            issues.append("Think block: " + prompt[:50])
        if not is_coherent:
            garbage_outputs += 1
            issues.append("Garbage output: " + prompt[:50])

        print(f"  Q: {prompt[:50]}...")
        print(f"  A: {resp[:150]}...")
        print(f"  [think={has_think}, coherent={is_coherent}]")
        print(flush=True)

    verdict = think_blocks == 0 and garbage_outputs <= 1

    print(f"\nLoRA SANITY RESULT: {model_name}")
    print(f"  Think blocks: {think_blocks}/10")
    print(f"  Garbage outputs: {garbage_outputs}/10")
    print(f"  Issues: {issues if issues else 'None'}")
    print(f"  VERDICT: {'PASS' if verdict else 'FAIL'}", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "passed": verdict,
        "think_blocks": think_blocks,
        "garbage_outputs": garbage_outputs,
        "issues": issues,
        "avg_train_loss": total_loss / 5,
    }


if __name__ == "__main__":
    os.makedirs("/workspace/persistent/eval", exist_ok=True)

    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    mode = sys.argv[2] if len(sys.argv) > 2 else "both"

    if model_name and model_name in MODELS:
        models_to_run = {model_name: MODELS[model_name]}
    elif model_name:
        print("Unknown model: " + model_name + ". Available: " + str(list(MODELS.keys())))
        sys.exit(1)
    else:
        models_to_run = MODELS

    all_results = {}

    for name, path in models_to_run.items():
        if not os.path.exists(path):
            print("SKIP " + name + ": " + path + " not found")
            continue

        result = {}

        if mode in ("humaneval", "both"):
            try:
                result["humaneval"] = run_humaneval(name, path)
            except Exception as e:
                print("ERROR in HumanEval for " + name + ": " + str(e))
                traceback.print_exc()
                result["humaneval"] = {"error": str(e)}

        if mode in ("lora", "both"):
            try:
                result["lora_sanity"] = run_lora_sanity(name, path)
            except Exception as e:
                print("ERROR in LoRA sanity for " + name + ": " + str(e))
                traceback.print_exc()
                result["lora_sanity"] = {"error": str(e)}

        all_results[name] = result

        with open("/workspace/persistent/eval/baseline_all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, res in all_results.items():
        he = res.get("humaneval", {})
        ls = res.get("lora_sanity", {})
        score = he.get("score", "ERR")
        think = he.get("think_blocks", "?")
        lora_pass = ls.get("passed", "?")
        lora_str = "PASS" if lora_pass is True else ("FAIL" if lora_pass is False else "?")
        print(f"  {name}: HumanEval={score}%, think_blocks={think}, LoRA_sanity={lora_str}")
    print("\nResults saved to /workspace/persistent/eval/baseline_all_results.json")
