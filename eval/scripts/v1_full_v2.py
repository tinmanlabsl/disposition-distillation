"""v1-full v2: same-task held-out with GOLD-CHECKLIST judge aligned to step16.

Fix from v1: cheap binary judge disagreed with gold_checklist (53%) because it
measures different criterion. We now build a gold checklist for each fresh
prompt (teacher generates required claims) and score with per-claim YES/NO,
matching eval_judge_rubric_v3.py and the step16 correctness labels exactly.

Reuses v1_full.py caches for P1 (fresh prompts) and P3 (step16 features).

Phases:
  P1   (reuse)    load fresh prompts cache
  P2   (NEW)      generate gold_checklist for fresh prompts
  P3   (reuse)    extract hs_last on step16 100
  P4   (reuse)    generate Gemma responses on fresh prompts + hs_last
  P5   (NEW)      per-claim YES/NO scoring -> coverage -> binary correct
  P6/P7           train gate on step16, predict on fresh, dual criterion
"""
import os, sys, json, time, re, asyncio, httpx
from pathlib import Path
import numpy as np
import torch

PERSIST = "/workspace/persistent"
RESULTS = f"{PERSIST}/dd-v2/eval/results"
OUT     = f"{RESULTS}/v1_full_v2.json"
Path(RESULTS).mkdir(parents=True, exist_ok=True)

if "OPENROUTER_API_KEY" not in os.environ:
    for line in open(f"{PERSIST}/.env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
OR_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MINIMAX_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "deepseek/deepseek-v3.2-exp"

# ===== P1: reuse fresh prompts cache =====
FRESH_PROMPTS_PATH = f"{RESULTS}/v1_full_fresh_prompts.json"
def load_fresh_prompts():
    d = json.load(open(FRESH_PROMPTS_PATH))
    print(f"[P1] reused {len(d['prompts'])} fresh prompts", flush=True)
    return d["prompts"]

# ===== P2: generate gold checklist for each fresh prompt =====
CHECKLIST_GEN_SYS = (
    "You are an expert French culinary instructor and eval designer. "
    "For a given chef troubleshooting question, list 3 to 5 specific canonical facts or actions "
    "that a correct response MUST include. These should be concrete, verifiable claims: "
    "temperatures, ratios, named techniques, causal mechanisms, or specific recovery steps. "
    "Do NOT include generic advice. Each claim should be short (under 15 words) and factual."
)
CHECKLIST_GEN_TEMPLATE = """Chef troubleshooting question:
{q}

Output 3 to 5 required canonical claims, one per line, no numbering or bullets.
Each line must be a concrete factual/technique claim needed to correctly answer the question.
Output ONLY the claim lines, nothing else."""

async def gen_checklist(client, sem, q, retries=3):
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.post(URL,
                    headers={"Authorization": f"Bearer {OR_KEY}"},
                    json={"model": JUDGE_MODEL,
                          "messages":[{"role":"system","content":CHECKLIST_GEN_SYS},
                                      {"role":"user","content":CHECKLIST_GEN_TEMPLATE.format(q=q)}],
                          "temperature":0.2, "max_tokens":400},
                    timeout=120)
                resp.raise_for_status()
                txt = resp.json()["choices"][0]["message"]["content"]
                lines = [re.sub(r"^\s*[-*\d.]+\s*","",l).strip().strip('"') for l in txt.split("\n")]
                claims = [l for l in lines if 6 < len(l) < 140]
                if 3 <= len(claims) <= 6:
                    return claims
                if len(claims) > 6:
                    return claims[:5]
                if len(claims) >= 2:
                    return claims
            except Exception as e:
                if attempt == retries-1:
                    return None
                await asyncio.sleep(2**attempt)
        return None

async def build_gold_for_fresh(client, prompts):
    path = f"{RESULTS}/v1_full_fresh_gold.json"
    if os.path.exists(path):
        g = json.load(open(path))
        print(f"[P2] cached gold: {len(g)} entries", flush=True)
        return g
    sem = asyncio.Semaphore(8)
    t0 = time.time()
    tasks = [gen_checklist(client, sem, p) for p in prompts]
    checklists = await asyncio.gather(*tasks)
    gold = []
    for i, (p, cl) in enumerate(zip(prompts, checklists)):
        if cl and len(cl) >= 3:
            gold.append({"i":i,"prompt":p,"required":cl})
    print(f"[P2] built gold for {len(gold)}/{len(prompts)} prompts (discarded {len(prompts)-len(gold)}) t={time.time()-t0:.0f}s", flush=True)
    if gold:
        avg = sum(len(g['required']) for g in gold) / len(gold)
        print(f"[P2] avg claims per prompt: {avg:.1f}", flush=True)
    json.dump(gold, open(path,"w"), indent=2)
    return gold

# ===== Per-claim judge (mirrors eval_judge_rubric_v3.py) =====
CLAIM_SYS = ('You are a strict French culinary expert judging whether a model response contains a specific factual claim. '
             'Reply with ONE word: YES or NO. Be literal. Synonyms and paraphrases count as YES. '
             'If the response only vaguely gestures at the idea without stating it, say NO.')
CLAIM_TEMPLATE = '''Question asked to the model:
{q}

Model response:
{r}

Required claim to check: "{claim}"

Does the response state this claim (or a clear paraphrase / synonym of it)? Reply YES or NO only.'''

async def ask_claim(client, sem, q, r, claim, retries=3):
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.post(URL,
                    headers={"Authorization":f"Bearer {OR_KEY}"},
                    json={"model":JUDGE_MODEL,
                          "messages":[{"role":"system","content":CLAIM_SYS},
                                      {"role":"user","content":CLAIM_TEMPLATE.format(q=q,r=r[:2500],claim=claim)}],
                          "temperature":0.0,"max_tokens":5},
                    timeout=120)
                resp.raise_for_status()
                txt = resp.json()["choices"][0]["message"]["content"].strip().upper()
                return 1 if "YES" in txt else 0
            except Exception:
                if attempt == retries-1:
                    return None
                await asyncio.sleep(2**attempt)

async def score_response(client, sem, q, r, claims):
    scores = await asyncio.gather(*[ask_claim(client, sem, q, r, c) for c in claims])
    valid = [s for s in scores if s is not None]
    if not valid: return None
    return sum(valid) / len(valid)

# ===== P3: step16 features — reuse cache from v1_full.py =====
def load_step16_features():
    path = f"{RESULTS}/v1_full_step16_features.npz"
    if not os.path.exists(path):
        print("[P3] ERROR: step16 features not cached; run v1_full.py first for P3", flush=True)
        sys.exit(4)
    d = np.load(path)
    print(f"[P3] reused step16 features X={d['X'].shape} y+={int(d['y'].sum())}", flush=True)
    return d["X"], d["y"]

# ===== P4: fresh responses + features — reuse cache =====
def load_fresh_responses():
    resp_path = f"{RESULTS}/v1_full_fresh_responses.json"
    feat_path = f"{RESULTS}/v1_full_fresh_features.npz"
    if not os.path.exists(resp_path) or not os.path.exists(feat_path):
        return None, None
    rows = json.load(open(resp_path))
    X = np.load(feat_path)["X"]
    print(f"[P4] reused fresh responses n={len(rows)} X={X.shape}", flush=True)
    return rows, X

# If not cached, actually generate (falls back to running model)
def run_fresh_generation(prompts):
    rows, X = load_fresh_responses()
    if rows is not None:
        return rows, X
    print(f"[P4] generating responses on {len(prompts)} prompts...", flush=True)
    from unsloth import FastModel
    m, t = FastModel.from_pretrained("unsloth/gemma-4-E2B-it", max_seq_length=2048,
                                      load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(m); m.eval()
    text_tok = t.tokenizer if hasattr(t,"tokenizer") else t
    def gen_one(p):
        msgs=[{"role":"user","content":[{"type":"text","text":p}]}]
        rendered = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        p_ids = text_tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            out = m.generate(p_ids, max_new_tokens=512, do_sample=False, use_cache=True,
                             pad_token_id=text_tok.pad_token_id or text_tok.eos_token_id)
        plen = p_ids.shape[1]; rlen = out.shape[1]-plen
        if rlen < 2: return None, None
        if rlen > 512:
            out = out[:, :plen+512]; rlen = 512
        with torch.no_grad():
            hsm = m(out, output_hidden_states=True)
        hs = hsm.hidden_states[-1][0]
        hs_last = hs[plen+rlen-1].float().cpu().numpy().astype(np.float32)
        resp = text_tok.decode(out[0, plen:plen+rlen], skip_special_tokens=True)
        return resp, hs_last
    rows, X = [], []
    t0 = time.time()
    for i, p in enumerate(prompts):
        try:
            r, h = gen_one(p)
        except Exception as e:
            print(f"  [{i}] err: {e}", flush=True); continue
        if r is None: continue
        rows.append({"i":i,"prompt":p,"response":r}); X.append(h)
        if (i+1)%20==0:
            print(f"  [{i+1}/{len(prompts)}] t={time.time()-t0:.0f}s", flush=True)
    X = np.stack(X)
    json.dump(rows, open(f"{RESULTS}/v1_full_fresh_responses.json","w"), indent=2)
    np.savez(f"{RESULTS}/v1_full_fresh_features.npz", X=X)
    print(f"[P4] done n={len(rows)} t={time.time()-t0:.0f}s", flush=True)
    del m; torch.cuda.empty_cache()
    return rows, X

# ===== P3 fallback: extract step16 features if not cached =====
def extract_step16_features():
    X, y = load_step16_features() if os.path.exists(f"{RESULTS}/v1_full_step16_features.npz") else (None, None)
    if X is not None: return X, y
    print("[P3] extracting step16 features fresh...", flush=True)
    from unsloth import FastModel
    m, t = FastModel.from_pretrained("unsloth/gemma-4-E2B-it", max_seq_length=2048,
                                      load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(m); m.eval()
    text_tok = t.tokenizer if hasattr(t,"tokenizer") else t
    base = json.load(open(f"{RESULTS}/baseline_gemma4_v2_from_step16.json"))["items"]
    X, y = [], []
    t0 = time.time()
    for i, it in enumerate(base):
        msgs=[{"role":"user","content":[{"type":"text","text":it['prompt']}]}]
        rendered = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        p_ids = text_tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids
        r_ids = text_tok(it['response'], return_tensors="pt", add_special_tokens=False).input_ids
        full = torch.cat([p_ids, r_ids], 1).to("cuda")
        plen = p_ids.shape[1]; rlen = r_ids.shape[1]
        if rlen < 2: continue
        if rlen > 512:
            full = full[:, :plen+512]; rlen = 512
        with torch.no_grad():
            out = m(full, output_hidden_states=True)
        hs = out.hidden_states[-1][0]
        X.append(hs[plen+rlen-1].float().cpu().numpy().astype(np.float32))
        y.append(1 if it["correct"] else 0)
        if (i+1) % 20 == 0:
            print(f"  step16 [{i+1}/{len(base)}] t={time.time()-t0:.0f}s", flush=True)
    X = np.stack(X); y = np.array(y)
    np.savez(f"{RESULTS}/v1_full_step16_features.npz", X=X, y=y)
    del m; torch.cuda.empty_cache()
    return X, y

# ===== P5: score fresh responses with gold checklist =====
async def judge_fresh_gold(client, rows, gold):
    path = f"{RESULTS}/v1_full_fresh_judged_gold.json"
    if os.path.exists(path):
        d = json.load(open(path))
        print(f"[P5] cached gold-judged n={len(d['coverage'])}", flush=True)
        return d["coverage"], d["labels"]
    gold_by_prompt = {g["prompt"]: g["required"] for g in gold}
    sem = asyncio.Semaphore(8)
    t0 = time.time()
    # Keep only rows whose prompt has a gold checklist
    work = [(i, r) for i, r in enumerate(rows) if r["prompt"] in gold_by_prompt]
    print(f"[P5] scoring {len(work)}/{len(rows)} fresh responses with gold checklists", flush=True)
    tasks = [score_response(client, sem, r["prompt"], r["response"], gold_by_prompt[r["prompt"]]) for _, r in work]
    covs = await asyncio.gather(*tasks)
    # Full-length coverage list aligned to rows (None where no gold)
    coverage = [None]*len(rows)
    for (idx, _), c in zip(work, covs):
        coverage[idx] = c
    labels = [1 if c is not None and c >= 0.5 else (0 if c is not None else None) for c in coverage]
    valid = [l for l in labels if l is not None]
    print(f"[P5] done: n_valid={len(valid)} correct_rate={sum(valid)/max(1,len(valid)):.3f} t={time.time()-t0:.0f}s", flush=True)
    json.dump({"coverage":coverage,"labels":labels}, open(path,"w"), indent=2)
    return coverage, labels

# ===== P6/P7: train + evaluate =====
def train_and_evaluate(X_train, y_train, X_fresh, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    y = np.array([l if l is not None else -1 for l in labels])
    mask = y >= 0
    Xf = X_fresh[mask]; yf = y[mask]
    print(f"[P6] train n={len(y_train)} (pos={int(y_train.sum())})  "
          f"fresh n={len(yf)} (pos={int(yf.sum())}, corr_rate={yf.mean():.3f})", flush=True)
    sc = StandardScaler().fit(X_train)
    # Compare LogReg C=1.0 and C=0.1 (both were strong in v1a)
    results = {}
    for C in [1.0, 0.1]:
        clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X_train), y_train)
        p = clf.predict_proba(sc.transform(Xf))[:, 1]
        auc = float(roc_auc_score(yf, p)) if len(np.unique(yf)) > 1 else None
        best=None; bal=None
        n_w=int((yf==0).sum()); n_c=int((yf==1).sum())
        for thr in np.linspace(0.05,0.95,91):
            f = (p > thr).astype(int)
            p_w = float(f[yf==0].mean()) if n_w else 0.0
            p_c = float(f[yf==1].mean()) if n_c else 0.0
            asym = p_w - p_c
            if best is None or asym < best["asym"]:
                best = {"thr":float(thr),"p_assert_wrong":p_w,"p_assert_correct":p_c,"asym":asym}
            if p_c >= 0.70 and (bal is None or asym < bal["asym"]):
                bal = {"thr":float(thr),"p_assert_wrong":p_w,"p_assert_correct":p_c,"asym":asym}
        f = (p > 0.5).astype(int)
        thr05 = {"thr":0.5,
                 "p_assert_wrong": float(f[yf==0].mean()) if n_w else 0.0,
                 "p_assert_correct": float(f[yf==1].mean()) if n_c else 0.0}
        thr05["asym"] = thr05["p_assert_wrong"] - thr05["p_assert_correct"]
        bal_s = "none" if bal is None else f"{bal['asym']:.3f}@{bal['thr']:.2f}"
        print(f"[P7] C={C}: AUC={auc}  min={best['asym']:.3f}@{best['thr']:.2f}  "
              f"thr0.5 asym={thr05['asym']:.3f}  bal(p_c>=0.7)={bal_s}", flush=True)
        results[f"C={C}"] = {"auc":auc,"min":best,"thr_0.5":thr05,"balanced_pc_0.70":bal}
    # Best balanced across C values
    bb = None
    for k, r in results.items():
        bal = r["balanced_pc_0.70"]
        if bal is None: continue
        if bb is None or bal["asym"] < bb[1]["asym"]:
            bb = (k, bal)
    gate_ok = bb is not None and bb[1]["asym"] <= -0.15 and any(
        r["auc"] is not None and r["auc"] >= 0.63 for r in results.values())
    print(f"[P7] BEST: {bb[0] if bb else 'none'} asym={bb[1]['asym'] if bb else 'na'}", flush=True)
    print(f"[P7] GATE (AUC>=0.63 AND balanced asym<=-0.15): {gate_ok}", flush=True)
    return {"configs":results,"best_balanced":{"config":bb[0],**bb[1]} if bb else None,
            "gate_pass":bool(gate_ok),
            "n_fresh":int(len(yf)),"n_fresh_pos":int(yf.sum()),"n_fresh_neg":int((yf==0).sum())}

# ===== Main =====
async def main():
    t_all = time.time()
    async with httpx.AsyncClient() as client:
        prompts = load_fresh_prompts()
        gold = await build_gold_for_fresh(client, prompts)
        if len(gold) < 80:
            print(f"[P2] ABORT: only {len(gold)} usable gold entries", flush=True); sys.exit(2)

        X_train, y_train = extract_step16_features()
        rows, X_fresh = run_fresh_generation(prompts)
        coverage, labels = await judge_fresh_gold(client, rows, gold)

        eval_result = train_and_evaluate(X_train, y_train, X_fresh, labels)

        summary = {
            "n_fresh_prompts": len(prompts),
            "n_fresh_responses": len(rows),
            "n_gold_checklists": len(gold),
            "evaluation": eval_result,
            "total_elapsed": time.time()-t_all,
        }
        json.dump(summary, open(OUT,"w"), indent=2, default=float)
        print(f"[v1_full_v2] DONE gate_pass={eval_result['gate_pass']} -> {OUT}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
