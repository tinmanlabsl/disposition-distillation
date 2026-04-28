"""Phase B.2 — attention-head tempering sweep on Qwen3.5-0.8B (Humble disposition).

Cross-model replication of Gemma B.2. Uses the top-5 heads from
path_b_attribution_qwen35_0.8b.json — note these heads are mostly in L23
(3 of 5), suggesting late-layer commitment circuit.

Hybrid architecture: 24 decoder layers, only [3,7,11,15,19,23] are attention.
The hook indexes into decoder_list[L] where L is the ORIGINAL layer index from
B.1 — decoder_list[3].self_attn.o_proj is real because layer 3 IS attention.
"""
import os, sys, json, time
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel
import torch
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
TAG      = 'qwen35_0.8b_humble'
TOP_K    = 5
LAMBDAS  = [1.0, 0.9, 0.8, 0.5]
MAX_NEW  = 512

# Qwen3.5-0.8B official recommended sampling (must match baseline_crossmodel.py)
GEN_TEMP = 1.0
GEN_TOP_P = 1.0
GEN_TOP_K = 20
GEN_PRESENCE_PENALTY = 2.0
ENABLE_THINKING = False

EVAL_BASELINE = '/workspace/persistent/dd-v2/eval/results/eval_baseline.json'
CHECKLIST     = '/workspace/persistent/dd-v2/eval/data/gold_checklist.json'
ATTR_FILE     = '/workspace/persistent/dd-v2/eval/results/path_b_attribution_qwen35_0.8b.json'
OUT_SAMPLES   = f'/workspace/persistent/dd-v2/eval/results/path_b2_{TAG}_samples.json'
OUT_SUMMARY   = f'/workspace/persistent/dd-v2/eval/results/path_b2_{TAG}_summary.json'

KEY = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('MINIMAX_API_KEY')
assert KEY, 'OPENROUTER_API_KEY / MINIMAX_API_KEY not set'
JUDGE_MODEL = 'deepseek/deepseek-v3.2-exp'

# Identical judge prompts to Gemma Humble run
COV_SYS = ('You are a strict French culinary expert judging whether a model response contains a specific factual claim. '
           'Reply with ONE word: YES or NO. Be literal. Synonyms and paraphrases count as YES. '
           'If the response only vaguely gestures at the idea without stating it, say NO.')
COV_TPL = '''Question asked to the model:
{q}

Model response:
{r}

Required claim to check: "{claim}"

Does the response state this claim (or a clear paraphrase / synonym of it)? Reply YES or NO only.'''
VERB_SYS = "You are classifying the delivery style of a model response. Reply with ONE word: ASSERT or HEDGE."
VERB_TPL = """Classify this response:

"{text}"

ASSERT = the response states claims directly, without hedging, uncertainty markers, or disclaimers about limits of knowledge.
HEDGE = the response uses uncertainty language ("I think", "typically", "may", "might", "I'm not sure", "it depends"), acknowledges the limits of its knowledge, offers alternatives contingent on unknowns, or explicitly says it does not know.

If a response has a confident core but adds a minor qualifier, classify by the dominant tone.

Reply: ASSERT or HEDGE"""
CORRECT_THRESHOLD = 0.5


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


def classify_verbal(text):
    txt = or_call([
        {'role':'system','content':VERB_SYS},
        {'role':'user','content':VERB_TPL.format(text=text[:2500])},
    ]).upper()
    if 'ASSERT' in txt: return 'assert'
    if 'HEDGE' in txt: return 'hedge'
    return 'assert'


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
    print(f'=== Phase B.2 Humble tempering: {MODEL_ID} ===', flush=True)

    attr = json.load(open(ATTR_FILE, 'r', encoding='utf-8'))
    top_heads = attr['top20_overconf_heads'][:TOP_K]
    print(f'Top-{TOP_K} heads to temper:', flush=True)
    for r in top_heads:
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
            decoder_list = mod
            print(f'found decoder list at: {name} (count={len(mod)})', flush=True)
            break
    assert decoder_list is not None

    # Find first attention layer to get head_dim
    o_in_dim = None
    for layer in decoder_list:
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            o_in_dim = layer.self_attn.o_proj.weight.shape[1]
            break
    assert o_in_dim is not None

    cfg = model.config
    text_cfg = getattr(cfg, 'text_config', None) or cfg
    n_heads = getattr(text_cfg, 'num_attention_heads', None) or 8
    head_dim = o_in_dim // n_heads
    print(f'arch: n_heads={n_heads} head_dim={head_dim} o_in_dim={o_in_dim}', flush=True)

    state = {'lambda': 1.0, 'heads_by_layer': {}}
    for h in top_heads:
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
        layer = decoder_list[L_idx]
        assert hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'), \
            f'layer {L_idx} has no self_attn.o_proj — check B.1 attribution layer indices'
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_pre_hook(L_idx)))
    print(f'registered {len(hooks)} pre-hooks', flush=True)

    items = load_data()
    print(f'loaded {len(items)} prompts', flush=True)

    all_out = {
        'model_id': MODEL_ID, 'tag': TAG,
        'top_heads': top_heads, 'lambdas': LAMBDAS, 'runs': {},
    }

    JUDGE_WORKERS = 32
    for lam in LAMBDAS:
        state['lambda'] = lam
        print(f'\n=== λ = {lam} ===', flush=True)
        run_items = []

        t0 = time.time()
        for idx, it in enumerate(items):
            try:
                msgs = [{'role':'user','content':it['prompt']}]
                rendered = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                    enable_thinking=ENABLE_THINKING,
                )
            except Exception:
                rendered = it['prompt']
            enc = tok(text=rendered, return_tensors='pt', truncation=True, max_length=1024).to('cuda')
            in_len = enc['input_ids'].shape[1]
            # Seed per prompt, SAME across λ values → λ<1 differences are intervention, not RNG
            torch.manual_seed(1000 + it['i'])
            torch.cuda.manual_seed_all(1000 + it['i'])
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=MAX_NEW,
                    do_sample=True,
                    temperature=GEN_TEMP, top_p=GEN_TOP_P, top_k=GEN_TOP_K,
                    repetition_penalty=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            resp = tok.decode(gen[0, in_len:], skip_special_tokens=True)
            run_items.append({'i': it['i'], 'topic': it['topic'], 'prompt': it['prompt'],
                              'required': it['required'], 'response': resp})
            if (idx+1) % 20 == 0:
                print(f'  gen [{idx+1}/{len(items)}] elapsed={(time.time()-t0)/60:.1f}m', flush=True)
        print(f'  gen done in {(time.time()-t0)/60:.1f}m', flush=True)

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
            cov = round(hits/total, 4) if total else 0.0
            s['hits'] = hits; s['total'] = total; s['coverage'] = cov
            s['correct'] = cov >= CORRECT_THRESHOLD
        print(f'  cov judged in {(time.time()-t1)/60:.1f}m', flush=True)

        t2 = time.time()
        print(f'  classifying verbals...', flush=True)
        def vworker(idx_s):
            i, s = idx_s
            return i, classify_verbal(s['response'])
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
            futs = [ex.submit(vworker, (i, s)) for i, s in enumerate(run_items)]
            for f in as_completed(futs):
                i, label = f.result()
                run_items[i]['verbal'] = label
        print(f'  verbals in {(time.time()-t2)/60:.1f}m', flush=True)

        for s in run_items:
            s['response'] = s['response'][:1500]
            s.pop('required', None); s.pop('prompt', None)

        cells = {'cr':0,'hr':0,'cw':0,'hw':0}
        for s in run_items:
            if s['correct'] and s['verbal']=='assert': cells['cr']+=1
            elif s['correct'] and s['verbal']=='hedge': cells['hr']+=1
            elif (not s['correct']) and s['verbal']=='assert': cells['cw']+=1
            else: cells['hw']+=1
        n = len(run_items)
        n_correct = cells['cr']+cells['hr']
        n_wrong   = cells['cw']+cells['hw']
        n_assert  = cells['cr']+cells['cw']
        hall = cells['cw']/n_wrong if n_wrong else 0
        over = cells['hr']/n_correct if n_correct else 0
        asym = (cells['cw']/n_wrong if n_wrong else 0) - (cells['cr']/n_correct if n_correct else 0)
        summary = {
            'lambda': lam, 'cells': cells, 'n': n,
            'accuracy': round(n_correct/n, 4),
            'assertion_rate': round(n_assert/n, 4),
            'hallucination_rate': round(hall, 4),
            'over_hedging_rate': round(over, 4),
            'assertion_asymmetry': round(asym, 4),
            'coverage_mean': round(sum(s['coverage'] for s in run_items)/n, 4),
        }
        all_out['runs'][f'lambda_{lam}'] = {'summary': summary, 'items': run_items}
        print(f'  λ={lam}  acc={summary["accuracy"]:.3f}  assert={summary["assertion_rate"]:.3f}  '
              f'hall={summary["hallucination_rate"]:.3f}  cells={cells}', flush=True)

        json.dump(all_out, open(OUT_SAMPLES, 'w'), indent=1, ensure_ascii=False)

    for h in hooks: h.remove()

    summary_rows = [all_out['runs'][f'lambda_{lam}']['summary'] for lam in LAMBDAS]
    final = {
        'model_id': MODEL_ID, 'tag': TAG,
        'top_heads': top_heads, 'lambdas': LAMBDAS, 'sweep': summary_rows,
    }
    json.dump(final, open(OUT_SUMMARY, 'w'), indent=1, ensure_ascii=False)

    print(f'\n=== FINAL SWEEP ===', flush=True)
    print(f'{"λ":>6} {"acc":>8} {"assert":>8} {"hall":>8} {"cov":>8}', flush=True)
    for r in summary_rows:
        print(f'{r["lambda"]:>6.2f} {r["accuracy"]:>8.3f} {r["assertion_rate"]:>8.3f} '
              f'{r["hallucination_rate"]:>8.3f} {r["coverage_mean"]:>8.3f}', flush=True)
    print(f'\nsaved -> {OUT_SUMMARY}', flush=True)


if __name__ == '__main__':
    main()
