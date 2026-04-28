"""v1-full (same-task held-out): frozen-base disposition gate generalization test.

Phases (each phase gates the next):
  P1. Generate ~200 fresh error_correction prompts (DeepSeek V3.2 via OpenRouter)
  P1b. Dedup against chef_prompts.jsonl (string similarity + shingle jaccard)
  P2. Validate cheap binary judge on step16 100 vs gold_checklist.json labels
      GATE: agreement >= 0.80
  P3. Extract hs_last features for step16 100 (training data for gate)
  P4. Generate Gemma 4 E2B responses on fresh prompts, capture hs_last
  P5. Score fresh responses with cheap binary judge (validated in P2)
  P6. Train gate (LogReg C=0.1, hs_last) on step16 100; predict on fresh
  P7. Report fresh AUC, balanced-threshold post-gate assertion_asymmetry
      GATE: AUC >= 0.63 AND balanced asym <= -0.15

Each phase saves its artifact so we can resume or inspect individually.
"""
import os, sys, json, time, re, asyncio, httpx
from pathlib import Path
import numpy as np
import torch

PERSIST = "/workspace/persistent"
RESULTS = f"{PERSIST}/dd-v2/eval/results"
LOGS    = f"{PERSIST}/dd-v2/eval/logs"
OUT     = f"{RESULTS}/v1_full.json"
Path(RESULTS).mkdir(parents=True, exist_ok=True)

# Load env (.env has MINIMAX_API_KEY = OpenRouter key; script supports both names)
if "OPENROUTER_API_KEY" not in os.environ:
    try:
        for line in open(f"{PERSIST}/.env"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    except Exception as e:
        print(f"[env] warn: {e}", flush=True)
OR_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MINIMAX_API_KEY")
assert OR_KEY, "no OpenRouter/Minimax key"
URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "deepseek/deepseek-v3.2-exp"

# ---------- P1: Generate fresh error_correction prompts ----------
FRESH_PROMPTS_PATH = f"{RESULTS}/v1_full_fresh_prompts.json"

SAMPLES = []  # loaded lazily
def load_samples():
    global SAMPLES
    if SAMPLES: return SAMPLES
    rows = [json.loads(l) for l in open(f"{PERSIST}/cuisine_test/chef_prompts.jsonl")]
    SAMPLES = [r for r in rows if r.get("category") == "error_correction"]
    return SAMPLES

async def gen_fresh_prompts(client, n_target=200):
    if os.path.exists(FRESH_PROMPTS_PATH):
        d = json.load(open(FRESH_PROMPTS_PATH))
        print(f"[P1] cached {len(d['prompts'])} prompts", flush=True)
        return d["prompts"]
    ex = load_samples()[:6]
    example_block = "\n".join(f"{i+1}. {e['prompt']}" for i, e in enumerate(ex))
    user = (
        f"Generate {n_target} NOVEL French culinary troubleshooting scenarios. "
        "Each scenario must:\n"
        " - describe a mid-execution failure a chef is experiencing (a sauce has split, a preparation has failed, a timing has gone wrong, a texture is off, etc.)\n"
        " - ask the model to diagnose the likely cause(s) and propose a concrete recovery or correction\n"
        " - be 1 to 4 sentences, written as the chef speaking in first or second person\n"
        " - cover distinct dishes/techniques — do not repeat the same failure mode\n"
        " - be comparable in style and difficulty to these reference examples:\n\n"
        f"{example_block}\n\n"
        f"Output EXACTLY {n_target} lines. Each line is one scenario. No numbering, no bullets, no blank lines, no commentary."
    )
    sys_msg = "You are an expert French cuisine instructor writing evaluation prompts. Be concrete, specific, and varied."
    resp = await client.post(URL,
        headers={"Authorization": f"Bearer {OR_KEY}"},
        json={"model": JUDGE_MODEL,
              "messages": [{"role": "system", "content": sys_msg},
                           {"role": "user", "content": user}],
              "temperature": 0.9, "max_tokens": 8000},
        timeout=180)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    lines = [l.strip() for l in text.split("\n")]
    prompts = [re.sub(r"^\s*\d+\.\s*", "", l).strip() for l in lines if len(l) > 40]
    prompts = [p for p in prompts if "?" in p or any(w in p.lower() for w in
               ["diagnose","fix","salvage","recover","correct","adjust","rescue","propose","how","what","why"])]
    print(f"[P1] generated {len(prompts)} candidate prompts", flush=True)
    json.dump({"prompts": prompts, "model": JUDGE_MODEL}, open(FRESH_PROMPTS_PATH, "w"), indent=2)
    return prompts

def shingle_set(s, k=6):
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return {s[i:i+k] for i in range(max(0, len(s)-k+1))}

def dedup_against_pool(prompts, pool_prompts, thr=0.35):
    pool_shingles = [shingle_set(p) for p in pool_prompts]
    kept, dropped = [], 0
    for p in prompts:
        ps = shingle_set(p)
        max_j = 0.0
        for ss in pool_shingles:
            if not ps or not ss: continue
            j = len(ps & ss) / max(1, len(ps | ss))
            if j > max_j: max_j = j
            if max_j > thr: break
        if max_j <= thr:
            kept.append(p)
        else:
            dropped += 1
    # Also dedup within kept
    kept2 = []
    seen_shingles = []
    for p in kept:
        ps = shingle_set(p)
        ok = True
        for ss in seen_shingles:
            if len(ps & ss) / max(1, len(ps | ss)) > 0.5:
                ok = False; break
        if ok:
            kept2.append(p); seen_shingles.append(ps)
    print(f"[P1b] dedup: kept {len(kept2)}/{len(prompts)} (dropped {dropped} vs pool, {len(kept)-len(kept2)} intra)", flush=True)
    return kept2

# ---------- P2: Validate cheap binary judge ----------
CHEAP_JUDGE_SYS = ("You are a strict French culinary expert. Evaluate whether a model's response correctly "
                   "answers the given chef question. Consider factual accuracy, technique correctness, "
                   "and whether the core issue in the question is addressed. Reply with ONE word: CORRECT or WRONG.")

CHEAP_JUDGE_TEMPLATE = """Chef question:
{q}

Model response:
{r}

Does the response correctly answer the question from a French culinary expert's perspective?
Reply CORRECT or WRONG only."""

async def cheap_judge(client, sem, q, r, retries=3):
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.post(URL,
                    headers={"Authorization": f"Bearer {OR_KEY}"},
                    json={"model": JUDGE_MODEL,
                          "messages": [{"role":"system","content":CHEAP_JUDGE_SYS},
                                       {"role":"user","content":CHEAP_JUDGE_TEMPLATE.format(q=q, r=r[:2500])}],
                          "temperature": 0.0, "max_tokens": 5},
                    timeout=120)
                resp.raise_for_status()
                txt = resp.json()["choices"][0]["message"]["content"].strip().upper()
                if "CORRECT" in txt and "WRONG" not in txt: return 1
                if "WRONG" in txt: return 0
                return None
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  judge err: {e}", flush=True); return None
                await asyncio.sleep(2**attempt)

async def validate_judge(client):
    path = f"{RESULTS}/v1_full_judge_validation.json"
    if os.path.exists(path):
        v = json.load(open(path))
        print(f"[P2] cached validation: agreement={v['agreement']:.3f}", flush=True)
        return v
    base = json.load(open(f"{RESULTS}/baseline_gemma4_v2_from_step16.json"))["items"]
    sem = asyncio.Semaphore(8)
    t0 = time.time()
    tasks = [cheap_judge(client, sem, it["prompt"], it["response"]) for it in base]
    preds = await asyncio.gather(*tasks)
    labels = [1 if it["correct"] else 0 for it in base]
    valid = [(p, l) for p, l in zip(preds, labels) if p is not None]
    agree = sum(1 for p, l in valid if p == l) / max(1, len(valid))
    from collections import Counter
    cm = Counter()
    for p, l in valid:
        cm[(p, l)] += 1
    result = {"n": len(valid), "agreement": agree,
              "confusion": {"pred=1|y=1": cm[(1,1)], "pred=0|y=0": cm[(0,0)],
                            "pred=1|y=0": cm[(1,0)], "pred=0|y=1": cm[(0,1)]},
              "elapsed": time.time()-t0}
    json.dump(result, open(path, "w"), indent=2)
    print(f"[P2] agreement={agree:.3f}  n={len(valid)}  cm={dict(result['confusion'])}  t={time.time()-t0:.0f}s", flush=True)
    return result

# ---------- P3/P4: Load model, extract features on step16 + fresh ----------
def load_gemma_4bit():
    from unsloth import FastModel
    m, t = FastModel.from_pretrained("unsloth/gemma-4-E2B-it", max_seq_length=2048,
                                      load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(m)
    m.eval()
    text_tok = t.tokenizer if hasattr(t, "tokenizer") else t
    return m, t, text_tok

def build_inputs(tok, text_tok, prompt, response=None):
    msgs = [{"role":"user","content":[{"type":"text","text":prompt}]}]
    rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p_ids = text_tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids
    if response is None:
        return p_ids, p_ids.shape[1], 0
    r_ids = text_tok(response, return_tensors="pt", add_special_tokens=False).input_ids
    return torch.cat([p_ids, r_ids], dim=1), p_ids.shape[1], r_ids.shape[1]

@torch.no_grad()
def feat_hslast_from_response(model, tok, text_tok, prompt, response):
    full, plen, rlen = build_inputs(tok, text_tok, prompt, response)
    if rlen < 2: return None
    if rlen > 512:
        full = full[:, :plen+512]; rlen = 512
    full = full.to("cuda")
    out = model(full, output_hidden_states=True)
    hs = out.hidden_states[-1][0]
    return hs[plen+rlen-1].float().cpu().numpy().astype(np.float32)

@torch.no_grad()
def generate_and_hslast(model, tok, text_tok, prompt, max_new=512):
    p_ids, plen, _ = build_inputs(tok, text_tok, prompt, None)
    p_ids = p_ids.to("cuda")
    gen = model.generate(p_ids, max_new_tokens=max_new, do_sample=False, use_cache=True,
                         pad_token_id=text_tok.pad_token_id or text_tok.eos_token_id)
    full = gen  # includes input
    rlen = full.shape[1] - plen
    if rlen < 2: return None, None
    if rlen > 512:
        full = full[:, :plen+512]; rlen = 512
    out = model(full, output_hidden_states=True)
    hs = out.hidden_states[-1][0]
    hs_last = hs[plen+rlen-1].float().cpu().numpy().astype(np.float32)
    response = text_tok.decode(full[0, plen:plen+rlen], skip_special_tokens=True)
    return response, hs_last

def extract_step16_features():
    path = f"{RESULTS}/v1_full_step16_features.npz"
    if os.path.exists(path):
        d = np.load(path)
        print(f"[P3] cached step16 features: X={d['X'].shape} y=+{int(d['y'].sum())}/-{int((d['y']==0).sum())}", flush=True)
        return d["X"], d["y"]
    print("[P3] extracting hs_last on step16 100 items...", flush=True)
    model, tok, text_tok = load_gemma_4bit()
    base = json.load(open(f"{RESULTS}/baseline_gemma4_v2_from_step16.json"))["items"]
    X, y = [], []
    t0 = time.time()
    for i, it in enumerate(base):
        h = feat_hslast_from_response(model, tok, text_tok, it["prompt"], it["response"])
        if h is None: continue
        X.append(h); y.append(1 if it["correct"] else 0)
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(base)}] t={time.time()-t0:.0f}s", flush=True)
    X = np.stack(X); y = np.array(y)
    np.savez(path, X=X, y=y)
    print(f"[P3] saved step16 features {X.shape}", flush=True)
    del model; torch.cuda.empty_cache()
    return X, y

def run_fresh_generation(prompts):
    path = f"{RESULTS}/v1_full_fresh_responses.json"
    feat_path = f"{RESULTS}/v1_full_fresh_features.npz"
    if os.path.exists(path) and os.path.exists(feat_path):
        rows = json.load(open(path))
        X = np.load(feat_path)["X"]
        print(f"[P4] cached fresh: {len(rows)} rows, X={X.shape}", flush=True)
        return rows, X
    print(f"[P4] generating on {len(prompts)} fresh prompts...", flush=True)
    model, tok, text_tok = load_gemma_4bit()
    rows, X = [], []
    t0 = time.time()
    for i, p in enumerate(prompts):
        try:
            resp, h = generate_and_hslast(model, tok, text_tok, p, max_new=512)
        except Exception as e:
            print(f"  [{i}] err: {e}", flush=True); continue
        if resp is None or h is None: continue
        rows.append({"i": i, "prompt": p, "response": resp})
        X.append(h)
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(prompts)}] t={time.time()-t0:.0f}s", flush=True)
    X = np.stack(X)
    json.dump(rows, open(path, "w"), indent=2)
    np.savez(feat_path, X=X)
    print(f"[P4] saved {len(rows)} responses + features {X.shape} in {time.time()-t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()
    return rows, X

# ---------- P5: Judge fresh ----------
async def judge_fresh(client, rows):
    path = f"{RESULTS}/v1_full_fresh_judged.json"
    if os.path.exists(path):
        d = json.load(open(path))
        print(f"[P5] cached judge labels n={len(d['labels'])}", flush=True)
        return d["labels"]
    sem = asyncio.Semaphore(8)
    t0 = time.time()
    tasks = [cheap_judge(client, sem, r["prompt"], r["response"]) for r in rows]
    labels = await asyncio.gather(*tasks)
    n_valid = sum(1 for l in labels if l is not None)
    n_correct = sum(1 for l in labels if l == 1)
    print(f"[P5] judged: valid={n_valid}/{len(labels)} correct_rate={n_correct/max(1,n_valid):.3f} t={time.time()-t0:.0f}s", flush=True)
    json.dump({"labels": labels}, open(path, "w"), indent=2)
    return labels

# ---------- P6/P7: Train gate, evaluate on fresh ----------
def train_and_evaluate(X_train, y_train, X_fresh, y_fresh_raw):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    y_fresh = np.array([l if l is not None else -1 for l in y_fresh_raw])
    mask = y_fresh >= 0
    Xf = X_fresh[mask]; yf = y_fresh[mask]
    print(f"[P6] train n={len(y_train)} (pos={int(y_train.sum())})  "
          f"fresh n={len(yf)} (pos={int(yf.sum())}, corr_rate={yf.mean():.3f})", flush=True)
    sc = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X_train), y_train)
    p = clf.predict_proba(sc.transform(Xf))[:, 1]
    auc = float(roc_auc_score(yf, p)) if len(np.unique(yf)) > 1 else None
    # Sweep thresholds, report min asym and balanced (p_assert_correct >= 0.70)
    best = None; bal = None
    n_w = int((yf==0).sum()); n_c = int((yf==1).sum())
    for thr in np.linspace(0.05, 0.95, 91):
        f = (p > thr).astype(int)
        p_w = float(f[yf==0].mean()) if n_w else 0.0
        p_c = float(f[yf==1].mean()) if n_c else 0.0
        asym = p_w - p_c
        if best is None or asym < best["asym"]:
            best = {"thr":float(thr),"p_assert_wrong":p_w,"p_assert_correct":p_c,"asym":asym}
        if p_c >= 0.70 and (bal is None or asym < bal["asym"]):
            bal = {"thr":float(thr),"p_assert_wrong":p_w,"p_assert_correct":p_c,"asym":asym}
    # thr 0.5 diagnostic
    f = (p > 0.5).astype(int)
    thr05 = {"thr":0.5,
             "p_assert_wrong": float(f[yf==0].mean()) if n_w else 0.0,
             "p_assert_correct": float(f[yf==1].mean()) if n_c else 0.0,
             "asym": float((f[yf==0].mean() if n_w else 0) - (f[yf==1].mean() if n_c else 0))}
    bal_str = "none" if bal is None else f"{bal['asym']:.3f}@{bal['thr']:.2f}"
    print(f"[P7] fresh AUC={auc}  min_asym={best['asym']:.3f}@{best['thr']:.2f}  "
          f"thr0.5 asym={thr05['asym']:.3f}  bal(p_c>=0.7)={bal_str}", flush=True)
    gate_ok = (auc is not None and auc >= 0.63) and (bal is not None and bal["asym"] <= -0.15)
    print(f"[P7] GATE: fresh AUC>=0.63 AND balanced asym<=-0.15  ->  {gate_ok}", flush=True)
    return {"auc": auc, "min": best, "thr_0.5": thr05, "balanced_pc_0.70": bal, "gate_pass": bool(gate_ok),
            "n_fresh": int(len(yf)), "n_fresh_pos": int(yf.sum()), "n_fresh_neg": int((yf==0).sum())}

# ---------- Main ----------
async def main():
    t_all = time.time()
    async with httpx.AsyncClient() as client:
        # P1
        prompts = await gen_fresh_prompts(client, n_target=200)
        pool = [json.loads(l)["prompt"] for l in open(f"{PERSIST}/cuisine_test/chef_prompts.jsonl")]
        prompts = dedup_against_pool(prompts, pool, thr=0.35)
        if len(prompts) < 80:
            print(f"[P1] ABORT: only {len(prompts)} fresh prompts survived dedup", flush=True); sys.exit(2)

        # P2
        val = await validate_judge(client)
        if val["agreement"] < 0.80:
            print(f"[P2] ABORT: judge agreement {val['agreement']:.3f} < 0.80 — cheap judge not reliable enough", flush=True)
            json.dump({"aborted":"judge_validation","judge_val":val}, open(OUT,"w"), indent=2)
            sys.exit(3)

        # P3
        X_train, y_train = extract_step16_features()

        # P4
        rows, X_fresh = run_fresh_generation(prompts)

        # P5
        labels = await judge_fresh(client, rows)

        # P6/P7
        eval_result = train_and_evaluate(X_train, y_train, X_fresh, labels)

        summary = {
            "n_fresh_prompts": len(prompts),
            "n_fresh_responses": len(rows),
            "judge_validation": val,
            "evaluation": eval_result,
            "total_elapsed": time.time()-t_all,
        }
        json.dump(summary, open(OUT,"w"), indent=2, default=float)
        print(f"[v1_full] DONE gate_pass={eval_result['gate_pass']} -> {OUT}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
