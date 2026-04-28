# Step 16f: Gold-anchored binary checklist judge.
# For each prompt, ask DeepSeek per required claim: does this response state X? YES/NO.
# Score = fraction of claims hit. Deterministic, length-invariant, style-invariant.
import os, json, asyncio, httpx, re

RESULTS = '/workspace/persistent/dd-v2/eval/results'
GOLD = '/workspace/persistent/dd-v2/eval/data/gold_checklist.json'
CONDS = ['baseline', 'sft_only', 'full_ddv2', 'full_ddv2_sysprompt']
OUT = f'{RESULTS}/judge_v3_checklist.json'

PARAMS = {
    'judge_model': 'deepseek/deepseek-v3.2-exp',
    'temperature': 0.0,
    'max_tokens': 5,
    'rubric_version': 'gold_checklist_v1',
    'concurrency': 8,
}
URL = 'https://openrouter.ai/api/v1/chat/completions'
KEY = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('MINIMAX_API_KEY')
assert KEY

SYSTEM = ('You are a strict French culinary expert judging whether a model response contains a specific factual claim. '
          'Reply with ONE word: YES or NO. Be literal. Synonyms and paraphrases count as YES. '
          'If the response only vaguely gestures at the idea without stating it, say NO.')

TEMPLATE = '''Question asked to the model:
{q}

Model response:
{r}

Required claim to check: "{claim}"

Does the response state this claim (or a clear paraphrase / synonym of it)? Reply YES or NO only.'''

async def ask(client, sem, q, r, claim):
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.post(URL, headers={'Authorization': f'Bearer {KEY}'}, json={
                    'model': PARAMS['judge_model'],
                    'messages': [
                        {'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': TEMPLATE.format(q=q, r=r[:2500], claim=claim)},
                    ],
                    'temperature': 0.0, 'max_tokens': PARAMS['max_tokens'],
                }, timeout=120)
                resp.raise_for_status()
                txt = resp.json()['choices'][0]['message']['content'].strip().upper()
                return 1 if 'YES' in txt else 0
            except Exception:
                if attempt == 2:
                    return None
                await asyncio.sleep(2 ** attempt)

async def score_response(client, sem, q, r, claims):
    results = await asyncio.gather(*[ask(client, sem, q, r, c) for c in claims])
    valid = [x for x in results if x is not None]
    if not valid:
        return None, results
    return sum(valid) / len(valid), results

async def main():
    gold = json.load(open(GOLD))
    gold_by_i = {g['i']: g for g in gold}

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(PARAMS['concurrency'])
        out = {'params': PARAMS, 'conditions': {}}

        for cname in CONDS:
            print(f'judging {cname}', flush=True)
            rows = json.load(open(f'{RESULTS}/eval_{cname}.json'))['rows']
            per_prompt = []
            tasks = []
            for i, row in enumerate(rows):
                g = gold_by_i.get(i)
                if not g:
                    continue
                tasks.append((i, row['prompt'], row['response'], g['required']))

            scores = await asyncio.gather(*[score_response(client, sem, q, r, c) for _, q, r, c in tasks])
            for (i, q, r, claims), (frac, hits) in zip(tasks, scores):
                per_prompt.append({'i': i, 'fraction': frac, 'hits': hits, 'n_claims': len(claims)})

            valid = [p['fraction'] for p in per_prompt if p['fraction'] is not None]
            avg = sum(valid) / max(1, len(valid))
            out['conditions'][cname] = {
                'avg_coverage': round(avg, 3),
                'n': len(valid),
                'per_prompt': per_prompt,
            }
            print(f'  {cname}: avg_coverage={avg:.3f} (n={len(valid)})', flush=True)

        json.dump(out, open(OUT, 'w'), indent=2)
        print(f'\nSAVED {OUT}')
        print(f'\n{"cond":25s} {"coverage":>10s}')
        for c in CONDS:
            print(f'{c:25s} {out["conditions"][c]["avg_coverage"]:>10.3f}')

if __name__ == '__main__':
    asyncio.run(main())
