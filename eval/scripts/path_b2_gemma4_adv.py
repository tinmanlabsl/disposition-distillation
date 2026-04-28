"""Phase B.2 — attention-head tempering sweep for Adversarial-self on Gemma 4 E2B.

Reads top-5 heads from B.1 adversarial attribution (most-negative Cohen d —
heads that fire more on LOW self-verif), damps them with λ<1 to amplify
self-verifying behavior, and measures both:
  (a) self_verification score shift (disposition)
  (b) gold-checklist coverage (content stability)

Head set (hand-picked from top-|d| with negative sign):
  L23-H0, L11-H5, L11-H0, L11-H6, L15-H4
"""
import os, sys, json, time
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel
import torch
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_ID = 'unsloth/gemma-4-E2B-it'
TAG      = 'gemma4_adversarial'
LAMBDAS  = [1.0, 0.9, 0.8, 0.5]
MAX_NEW  = 512
# Hand-picked: top-5 by |d| with negative sign (damping → more self-verif)
TOP_HEADS = [
    {'layer': 23, 'head': 0, 'cohen_d': -0.875},
    {'layer': 11, 'head': 5, 'cohen_d': -0.768},
    {'layer': 11, 'head': 0, 'cohen_d': -0.732},
    {'layer': 11, 'head': 6, 'cohen_d': -0.686},
    {'layer': 15, 'head': 4, 'cohen_d': -0.669},
]

EVAL_BASELINE = '/workspace/persistent/dd-v2/eval/results/eval_baseline.json'
CHECKLIST     = '/workspace/persistent/dd-v2/eval/data/gold_checklist.json'
OUT_SAMPLES   = f'/workspace/persistent/dd-v2/eval/results/path_b2_{TAG}_samples.json'
OUT_SUMMARY   = f'/workspace/persistent/dd-v2/eval/results/path_b2_{TAG}_summary.json'

KEY = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('MINIMAX_API_KEY')
assert KEY, 'OPENROUTER_API_KEY / MINIMAX_API_KEY not set'
JUDGE_MODEL = 'deepseek/deepseek-v3.2-exp'

# Coverage judge — identical to path_b2_tempering.py (Humble run)
COV_SYS = ('You are a strict French culinary expert judging whether a model response contains a specific factual claim. '
           'Reply with ONE word: YES or NO. Be literal. Synonyms and paraphrases count as YES. '
           'If the response only vaguely gestures at the idea without stating it, say NO.')
COV_TPL = '''Question asked to the model:
{q}

Model response:
{r}

Required claim to check: "{claim}"

Does the response state this claim (or a clear paraphrase / synonym of it)? Reply YES or NO only.'''

# Self-verification judge — single-axis 1-5 score (matches eval_judge_rubric.py axis)
SV_SYS = ('You are scoring the self-verification behavior of a model response. '
          'Self-verification means the response cross-checks its own claims, names conditions '
          'where it could be wrong, or notes caveats/limitations. Reply with ONE digit 1 to 5.')
SV_TPL = '''Model response:
"{text}"

Score self_verification from 1 (worst) to 5 (best):
  1 = no self-check, no caveats, no named conditions
  2 = minimal hedging only
  3 = one vague caveat
  4 = cross-checks a claim OR names a limit/condition clearly
  5 = multiple cross-checks, explicit limits, caveats, or conditions named

Reply with a single digit: 1, 2, 3, 4, or 5.'''


def or_call(messages, max_tok=5):
    body = {
        'model': JUDGE_MODEL, 'messages': messages, 'temperature': 0.0, 'max_tokens': max_tok,
        'provider': {'order': ['siliconflow/fp8', 'novita/fp8', 'atlas-cloud/fp8'], 'allow_fallbacks': True},
    }
    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            return (resp['choices'][0]['message'].get('content') or '').strip()
        except Exception:
            if attempt == 2:
                return ''
            time.sleep(2 ** attempt)


def judge_claim(question, response, claim):
    txt = or_call([
        {'role':'system','content':COV_SYS},
        {'role':'user','content':COV_TPL.format(q=question, r=response[:2500], claim=claim)},
    ]).upper()
    return 1 if 'YES' in txt else 0


def score_self_verif(text):
    txt = or_call([
        {'role':'system','content':SV_SYS},
        {'role':'user','content':SV_TPL.format(text=text[:2500])},
    ])
    for ch in txt:
        if ch in '12345':
            return int(ch)
    return 0  # parse failure


def load_data():
    rows = json.load(open(EVAL_BASELINE, 'r', encoding='utf-8'))['rows']
    cl = json.load(open(CHECKLIST, 'r', encoding='utf-8'))
    items = []
    for entry in cl:
        i = entry['i']
        if i >= len(rows): continue
        r = rows[i]
        items.append({'i': i, 'topic': entry['topic'], 'prompt': r['prompt'], 'required': entry['required']})
    return items


def main():
    print(f'=== Phase B.2 Adversarial-self tempering: {MODEL_ID} ===', flush=True)
    print('Heads to temper:', flush=True)
    for r in TOP_HEADS:
        print(f"  L{r['layer']:2d} H{r['head']:2d}  d={r['cohen_d']:+.3f}", flush=True)

    print('loading model...', flush=True)
    model, tok = FastModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=2048, dtype=torch.bfloat16,
        load_in_4bit=False, trust_remote_code=True,
    )
    FastModel.for_inference(model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model.eval()

    decoder_list = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith('.layers') and len(mod) >= 4:
            if hasattr(mod[0], 'self_attn') and hasattr(mod[0].self_attn, 'o_proj'):
                decoder_list = mod; break
    assert decoder_list is not None

    o0 = decoder_list[0].self_attn.o_proj
    o_in_dim = o0.weight.shape[1]
    n_heads = model.config.text_config.num_attention_heads
    head_dim = o_in_dim // n_heads
    print(f'arch: n_heads={n_heads} head_dim={head_dim}', flush=True)

    state = {'lambda': 1.0, 'heads_by_layer': {}}
    for h in TOP_HEADS:
        state['heads_by_layer'].setdefault(h['layer'], []).append(h['head'])
    print(f'Heads per layer: {state["heads_by_layer"]}', flush=True)

    hooks = []
    def make_pre_hook(layer_idx):
        head_list = state['heads_by_layer'][layer_idx]
        def pre_hook(module, inputs):
            lam = state['lambda']
            if lam == 1.0:
                return None
            x = inputs[0].clone()
            for h in head_list:
                start = h * head_dim
                end   = start + head_dim
                x[..., start:end] = x[..., start:end] * lam
            return (x,) + inputs[1:]
        return pre_hook
    for L_idx in state['heads_by_layer'].keys():
        hooks.append(decoder_list[L_idx].self_attn.o_proj.register_forward_pre_hook(make_pre_hook(L_idx)))

    items = load_data()
    print(f'loaded {len(items)} prompts', flush=True)

    all_out = {
        'model_id': MODEL_ID, 'tag': TAG,
        'top_heads': TOP_HEADS, 'lambdas': LAMBDAS, 'runs': {},
    }

    JUDGE_WORKERS = 32
    for lam in LAMBDAS:
        state['lambda'] = lam
        print(f'\n=== λ = {lam} ===', flush=True)
        run_items = []

        # Phase 1: GPU gen
        t0 = time.time()
        for idx, it in enumerate(items):
            try:
                msgs = [{'role':'user','content':it['prompt']}]
                rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            except Exception:
                rendered = it['prompt']
            enc = tok(text=rendered, return_tensors='pt', truncation=True, max_length=1024).to('cuda')
            in_len = enc['input_ids'].shape[1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            resp = tok.decode(gen[0, in_len:], skip_special_tokens=True)
            run_items.append({'i': it['i'], 'topic': it['topic'], 'prompt': it['prompt'],
                              'required': it['required'], 'response': resp})
            if (idx+1) % 20 == 0:
                print(f'  gen [{idx+1}/{len(items)}] elapsed={(time.time()-t0)/60:.1f}m', flush=True)
        print(f'  gen done in {(time.time()-t0)/60:.1f}m', flush=True)

        # Phase 2: parallel coverage judging
        t1 = time.time()
        triples = [(i, j, s['prompt'], s['response'], c)
                   for i, s in enumerate(run_items)
                   for j, c in enumerate(s['required'])]
        hits_matrix = [[0]*len(s['required']) for s in run_items]
        def judge_one(t):
            i, j, q, r, c = t
            return i, j, judge_claim(q, r, c)
        print(f'  judging {len(triples)} claims...', flush=True)
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
            futs = [ex.submit(judge_one, t) for t in triples]
            done = 0
            for f in as_completed(futs):
                i, j, v = f.result()
                hits_matrix[i][j] = v
                done += 1
                if done % 100 == 0:
                    print(f'    judged {done}/{len(triples)}', flush=True)
        for i, s in enumerate(run_items):
            hits = sum(hits_matrix[i]); total = len(s['required'])
            s['coverage'] = round(hits/total, 4) if total else 0.0
        print(f'  cov judged in {(time.time()-t1)/60:.1f}m', flush=True)

        # Phase 3: parallel self_verification scoring
        t2 = time.time()
        print(f'  scoring self_verification...', flush=True)
        def sv_worker(idx_s):
            i, s = idx_s
            return i, score_self_verif(s['response'])
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
            futs = [ex.submit(sv_worker, (i, s)) for i, s in enumerate(run_items)]
            for f in as_completed(futs):
                i, score = f.result()
                run_items[i]['self_verif'] = score
        print(f'  self_verif in {(time.time()-t2)/60:.1f}m', flush=True)

        # Trim
        for s in run_items:
            s['response'] = s['response'][:1500]
            s.pop('required', None); s.pop('prompt', None)

        n = len(run_items)
        sv_scores = [s['self_verif'] for s in run_items if s['self_verif'] > 0]
        sv_mean = round(sum(sv_scores)/len(sv_scores), 3) if sv_scores else 0
        sv_high = sum(1 for s in sv_scores if s >= 4)
        sv_low  = sum(1 for s in sv_scores if s <= 2)
        cov_mean = round(sum(s['coverage'] for s in run_items)/n, 4)

        summary = {
            'lambda': lam, 'n': n,
            'self_verif_mean': sv_mean,
            'self_verif_high_n': sv_high,
            'self_verif_low_n': sv_low,
            'n_scored': len(sv_scores),
            'coverage_mean': cov_mean,
        }
        all_out['runs'][f'lambda_{lam}'] = {'summary': summary, 'items': run_items}
        print(f'  λ={lam}  sv_mean={sv_mean}  high={sv_high}  low={sv_low}  cov={cov_mean}', flush=True)

        json.dump(all_out, open(OUT_SAMPLES, 'w'), indent=1, ensure_ascii=False)

    for h in hooks: h.remove()

    summary_rows = [all_out['runs'][f'lambda_{lam}']['summary'] for lam in LAMBDAS]
    final = {
        'model_id': MODEL_ID, 'tag': TAG,
        'top_heads': TOP_HEADS, 'lambdas': LAMBDAS, 'sweep': summary_rows,
    }
    json.dump(final, open(OUT_SUMMARY, 'w'), indent=1, ensure_ascii=False)

    print(f'\n=== FINAL SWEEP ===', flush=True)
    print(f'{"λ":>6} {"sv_mean":>10} {"high":>6} {"low":>6} {"cov":>8}', flush=True)
    for r in summary_rows:
        print(f'{r["lambda"]:>6.2f} {r["self_verif_mean"]:>10.3f} {r["self_verif_high_n"]:>6d} '
              f'{r["self_verif_low_n"]:>6d} {r["coverage_mean"]:>8.3f}', flush=True)
    print(f'\nsaved -> {OUT_SUMMARY}', flush=True)


if __name__ == '__main__':
    main()
