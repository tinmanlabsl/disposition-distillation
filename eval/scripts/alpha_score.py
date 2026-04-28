"""Score alpha_sweep.json outputs against gold checklist via DeepSeek V3.2 binary judge.
Also compute telemetry aggregates: mean KL, base-confidence buckets, divergence histogram."""
import os, json, time, statistics
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

KEY = os.environ.get('MINIMAX_API_KEY')  # OR key
assert KEY
SWEEP = '/workspace/persistent/dd-v2/eval/results/alpha_sweep.json'
GOLD  = '/workspace/persistent/dd-v2/eval/data/gold_checklist.json'
OUT   = '/workspace/persistent/dd-v2/eval/results/alpha_sweep_scored.json'
MODEL = 'deepseek/deepseek-v3.2-exp'

SYS = "You are a strict French culinary expert judging whether a model response contains a specific factual claim. Reply with ONE word: YES or NO. Be literal. Synonyms and paraphrases count as YES."

def judge(response, claim):
    body = {
        'model': MODEL,
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
                print(f'  judge err: {e}')
                return 0
            time.sleep(2)

def score_response(response, claims):
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(judge, response, c) for c in claims]
        hits = sum(f.result() for f in as_completed(futs))
    return hits, len(claims)

def telemetry_stats(tele):
    if not tele:
        return {}
    kls = [t['kl'] for t in tele]
    top1 = [t['base_top1'] for t in tele]
    margins = [t['margin'] for t in tele]
    # buckets by base confidence
    buckets = {'high': [], 'mid': [], 'low': []}  # high: top1>=0.7, mid 0.3-0.7, low <0.3
    for t in tele:
        b = 'high' if t['base_top1'] >= 0.7 else ('mid' if t['base_top1'] >= 0.3 else 'low')
        buckets[b].append(t['kl'])
    return {
        'n_tok': len(tele),
        'mean_kl': round(statistics.mean(kls), 4),
        'mean_base_top1': round(statistics.mean(top1), 4),
        'mean_margin': round(statistics.mean(margins), 4),
        'kl_by_conf': {k: (round(statistics.mean(v),4) if v else None, len(v)) for k,v in buckets.items()},
    }

def main():
    sweep = json.load(open(SWEEP))
    gold = {g['i']: g for g in json.load(open(GOLD))}
    out = {'alphas': sweep['alphas'], 'per_item': [], 'summary_by_alpha': {}}
    alpha_hits = {str(a): [] for a in sweep['alphas']}

    for item in sweep['items']:
        tag = item['tag']
        rec = {'tag': tag, 'by_alpha': {}}
        idx = None
        if tag.startswith('gold_'):
            idx = int(tag.split('_')[1])
            claims = gold[idx]['required']
        else:
            claims = None
        for a_str, payload in item['by_alpha'].items():
            text = payload['text']
            stats = telemetry_stats(payload.get('telemetry', []))
            r = {'n_tok': payload.get('n_tok'), 'telemetry': stats, 'text_preview': text[:400]}
            if claims:
                hits, total = score_response(text, claims)
                r['hits'] = hits
                r['total'] = total
                r['coverage'] = round(hits/total, 4)
                alpha_hits[a_str].append((hits, total))
                print(f'  {tag} a={a_str}: {hits}/{total} ({hits/total:.2f}) kl={stats.get("mean_kl")}', flush=True)
            else:
                r['coverage'] = None
                print(f'  {tag} a={a_str}: (novel, no gold) kl={stats.get("mean_kl")}', flush=True)
            rec['by_alpha'][a_str] = r
        out['per_item'].append(rec)

    for a_str, hs in alpha_hits.items():
        if hs:
            total_h = sum(h for h,_ in hs)
            total_t = sum(t for _,t in hs)
            out['summary_by_alpha'][a_str] = {'coverage': round(total_h/total_t, 4), 'hits': total_h, 'total': total_t, 'n_prompts': len(hs)}

    json.dump(out, open(OUT, 'w'), indent=1, ensure_ascii=False)
    print('\n=== SUMMARY ===')
    for a_str, s in sorted(out['summary_by_alpha'].items(), key=lambda x: float(x[0])):
        print(f'  a={a_str}: coverage {s["coverage"]} ({s["hits"]}/{s["total"]} across {s["n_prompts"]} prompts)')

if __name__ == '__main__':
    main()
