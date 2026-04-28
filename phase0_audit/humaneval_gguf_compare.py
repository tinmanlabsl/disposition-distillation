#!/usr/bin/env python -u
"""Apples-to-apples HumanEval comparison across GGUF quantizations.
All models run through llama-server with identical generation params."""

import json, os, re, subprocess, sys, time, urllib.request

MODELS = {
    "FP16": "/workspace/persistent/models/tinman-code-0.6B-f16.gguf",
    "Q4_K_M": "/workspace/persistent/models/tinman-code-0.6B-q4km.gguf",
    "Q5_K_M": "/workspace/persistent/models/tinman-code-0.6B-q5km.gguf",
}
SERVER = "/workspace/persistent/llama.cpp/build/bin/llama-server"
PORT = 8080

def start_server(model_path):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib64:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen(
        [SERVER, "-m", model_path, "-ngl", "99", "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )
    for _ in range(30):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return proc
        except: pass
    raise RuntimeError(f"Server failed to start for {model_path}")

def stop_server(proc):
    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(2)

def generate(prompt):
    chat = (
        "<|im_start|>user\n"
        "Complete the following Python function. Return ONLY the complete function, no explanation:\n\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    payload = json.dumps({
        "prompt": chat, "n_predict": 512, "temperature": 0,
        "top_k": -1, "top_p": 1.0,
        "stop": ["<|im_end|>", "<|endoftext|>"]
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read()).get("content", "").strip()

def extract_code(response, entry_point):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks: return blocks[0].strip()
    lines, code, in_code = response.split("\n"), [], False
    for l in lines:
        if l.strip().startswith("def ") or in_code:
            in_code = True; code.append(l)
            if in_code and l.strip() == "" and len(code) > 2: break
    return "\n".join(code).strip() if code else response.strip()

def run_test(problem, code):
    import tempfile
    entry = problem["entry_point"]
    if f"def {entry}" in code:
        full = code + "\n\n" + problem["test"] + f"\n\ncheck({entry})\n"
    else:
        full = problem["prompt"] + code + "\n\n" + problem["test"] + f"\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full); f.flush(); tmp = f.name
    try:
        r = subprocess.run(["python", tmp], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except: return False
    finally: os.unlink(tmp)

def main():
    from datasets import load_dataset
    ds = list(load_dataset("openai/openai_humaneval", split="test",
                           cache_dir="/workspace/persistent/models/hf_cache"))
    print(f"Loaded {len(ds)} HumanEval problems")
    results = {}

    for name, path in MODELS.items():
        print(f"\n=== {name} ({os.path.basename(path)}) ===")
        proc = start_server(path)
        passed = 0
        for i, p in enumerate(ds):
            resp = generate(p["prompt"])
            code = extract_code(resp, p["entry_point"])
            if run_test(p, code): passed += 1
            if (i+1) % 20 == 0:
                print(f"  {i+1}/164: {passed}/{i+1} passed ({passed/(i+1)*100:.1f}%)")
        score = passed / len(ds) * 100
        results[name] = {"passed": passed, "total": len(ds), "score": round(score, 1)}
        print(f"  FINAL: {passed}/164 = {score:.1f}%")
        stop_server(proc)

    print("\n" + "="*50)
    print("APPLES-TO-APPLES HUMANEVAL COMPARISON")
    print("="*50)
    print(f"{Model:<12} {Passed:>8} {Score:>8}")
    for name, r in results.items():
        print(f"{name:<12} {r[passed]:>5}/164 {r[score]:>7.1f}%")

    fp16 = results["FP16"]["score"]
    for name in ["Q4_K_M", "Q5_K_M"]:
        delta = fp16 - results[name]["score"]
        print(f"  {name} degradation vs FP16: {delta:.1f}%")

    with open("/workspace/persistent/eval/humaneval_gguf_compare.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /workspace/persistent/eval/humaneval_gguf_compare.json")

if __name__ == "__main__":
    main()
