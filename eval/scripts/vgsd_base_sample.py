"""VGSD feasibility probe: does base Gemma 4 E2B 'sometimes know it'?

For each of the 100 gold-checklist prompts, generate 8 samples at temperature=0.9
on the BASE model (no LoRA), score each sample against its gold checklist via
DeepSeek V3.2 binary judge. Output: per-prompt best/worst/mean coverage and the
fraction of prompts with at least one good sample (coverage >= 0.5).

This is pure diagnostic — no training, no commits. Tells us:
- Coverage ceiling for VGSD (fraction of prompts where best-of-N works)
- DPO signal strength (best - worst per prompt)
- Whether the 'base knows it sometimes' hypothesis holds empirically
"""
import os, json, time, statistics
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from unsloth import FastModel
import torch
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ADAPTER   = '/workspace/persistent/dd-v2/checkpoints/chef_dpo/final'  # disable_adapter() => base
GOLD_PATH = '/workspace/persistent/data/type1_gold.jsonl'
CHECKLIST = '/workspace/persistent/dd-v2/eval/data/gold_checklist.json'
OUT_PATH  = '/workspace/persistent/dd-v2/eval/results/vgsd_base_samples.json'
N_SAMPLES = 8
MAX_NEW   = 400
TEMP      = 0.9
TOP_P     = 0.95

KEY = os.environ.get('MINIMAX_API_KEY')
assert KEY, 'MINIMAX_API_KEY not set'
JUDGE_MODEL = 'deepseek/deepseek-v3.2-exp'
SYS = "You are a strict French culinary expert judging whether a model response contains a specific factual claim. Reply with ONE word: YES or NO. Be literal. Synonyms and paraphrases count as YES."

def judge(response, claim):
    body = {
        'model': JUDGE_MODEL,
        'messages': [
            {'role':'system','content':SYS},
            {'role':'user','content':f'CLAIM: {claim}\n\nRESPONSE:\n{response}\n\nDoes the response contain the claim? YES or NO.'},
        ],
        'temperature': 0.0,
        'max_tokens': 5,
    }
    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Authorization': f'Bearer {KEY}', 'Content-Type':'application/json'},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            txt = (resp['choices'][0]['message'].get('content') or '').strip().upper()
            return 1 if txt.startswith('Y') else 0
        except Exception as e:
            if attempt == 2:
                return 0
            time.sleep(2)

def score_response(response, claims):
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(judge, response, c) for c in claims]
        return sum(f.result() for f in as_completed(futs)), len(claims)

def load_data():
    lines = open(GOLD_PATH, 'r', encoding='utf-8').read().splitlines()
    cl = json.load(open(CHECKLIST, 'r', encoding='utf-8'))
    items = []
    for entry in cl:
        i = entry['i']
        if i >= len(lines):
            continue
        r = json.loads(lines[i])
        prompt = r.get('prompt') or r.get('question') or r.get('instruction')
        items.append({'i': i, 'topic': entry['topic'], 'prompt': prompt, 'required': entry['required']})
    return items

def main():
    print(f'loading model+adapter (will use disable_adapter for base)...', flush=True)
    m, t = FastModel.from_pretrained(ADAPTER, max_seq_length=2048, load_in_4bit=True, full_finetuning=False)
    FastModel.for_inference(m)
    text_tok = t.tokenizer if hasattr(t, 'tokenizer') else t
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    items = load_data()
    print(f'loaded {len(items)} prompts', flush=True)

    out = {'config': {'n_samples': N_SAMPLES, 'temp': TEMP, 'top_p': TOP_P, 'max_new': MAX_NEW}, 'items': []}
    t0 = time.time()
    for idx, it in enumerate(items):
        msgs = [{'role':'user','content':[{'type':'text','text':it['prompt']}]}]
        rendered = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = text_tok(rendered, return_tensors='pt', truncation=True, max_length=1024).to('cuda')
        in_len = enc['input_ids'].shape[1]

        samples = []
        with torch.no_grad(), m.disable_adapter():
            for s in range(N_SAMPLES):
                gen = m.generate(
                    **enc, max_new_tokens=MAX_NEW, do_sample=True,
                    temperature=TEMP, top_p=TOP_P,
                    pad_token_id=text_tok.pad_token_id,
                )
                text = text_tok.decode(gen[0, in_len:], skip_special_tokens=True)
                samples.append(text)

        scored = []
        for s_text in samples:
            hits, total = score_response(s_text, it['required'])
            cov = round(hits/total, 4) if total else 0.0
            scored.append({'text': s_text[:1500], 'hits': hits, 'total': total, 'coverage': cov})

        covs = [s['coverage'] for s in scored]
        rec = {
            'i': it['i'],
            'topic': it['topic'],
            'best_coverage': max(covs),
            'worst_coverage': min(covs),
            'mean_coverage': round(sum(covs)/len(covs), 4),
            'spread': round(max(covs) - min(covs), 4),
            'has_good_sample': max(covs) >= 0.5,
            'samples': scored,
        }
        out['items'].append(rec)

        elapsed = time.time() - t0
        eta = elapsed / (idx+1) * (len(items) - idx - 1)
        print(f'[{idx+1}/{len(items)}] i={it["i"]:3d} {it["topic"][:40]:40s} best={rec["best_coverage"]:.2f} worst={rec["worst_coverage"]:.2f} spread={rec["spread"]:.2f} good={rec["has_good_sample"]} | elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m', flush=True)

        # Save incrementally every 10 prompts
        if (idx+1) % 10 == 0:
            json.dump(out, open(OUT_PATH, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

    # Final aggregate
    n = len(out['items'])
    n_good = sum(1 for r in out['items'] if r['has_good_sample'])
    mean_best = statistics.mean(r['best_coverage'] for r in out['items'])
    mean_worst = statistics.mean(r['worst_coverage'] for r in out['items'])
    mean_spread = statistics.mean(r['spread'] for r in out['items'])
    full_coverage = sum(1 for r in out['items'] if r['best_coverage'] >= 1.0) / n

    out['summary'] = {
        'n_prompts': n,
        'n_with_good_sample': n_good,
        'frac_with_good_sample': round(n_good/n, 4),
        'mean_best_of_8': round(mean_best, 4),
        'mean_worst_of_8': round(mean_worst, 4),
        'mean_spread': round(mean_spread, 4),
        'frac_full_coverage': round(full_coverage, 4),
    }
    json.dump(out, open(OUT_PATH, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\n=== VGSD FEASIBILITY SUMMARY ===')
    print(json.dumps(out['summary'], indent=1))
    print(f'saved -> {OUT_PATH}')

if __name__ == '__main__':
    main()
