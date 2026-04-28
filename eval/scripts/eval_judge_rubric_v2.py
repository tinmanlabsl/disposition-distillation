# Step 16e: Two-pass rubric judge with response-blind certainty classification.
# Pass 1 — judge classifies each prompt CERTAIN/UNCERTAIN/AMBIGUOUS from the prompt ALONE.
# Pass 2 — each response scored against the FIXED certainty label, removing response contamination.
import os, json, asyncio, httpx, re

RESULTS = '/workspace/persistent/dd-v2/eval/results'
CONDS = ['baseline', 'sft_only', 'full_ddv2', 'full_ddv2_sysprompt']
CERT_FILE = f'{RESULTS}/cert_labels.json'
OUT_FILE  = f'{RESULTS}/judge_rubric_v2.json'

JUDGE_PARAMS = {
    'judge_model': 'deepseek/deepseek-v3.2-exp',
    'temperature': 0.0,
    'rubric_version': 'two_pass_response_blind_v1',
    'pass1_max_tokens': 30,
    'pass2_max_tokens': 300,
    'concurrency': 5,
}
JUDGE = JUDGE_PARAMS['judge_model']
URL = 'https://openrouter.ai/api/v1/chat/completions'
KEY = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('MINIMAX_API_KEY')
assert KEY

CERT_SYSTEM = ('You are a French culinary expert. Classify questions about French cuisine '
               'by how deterministic their answer is. Be strict.')
CERT_TEMPLATE = '''Question:
{q}

Without seeing any response, classify this question:
- CERTAIN: there is one canonical, widely-accepted correct answer in classical French cuisine
- UNCERTAIN: there are multiple valid answers / depends on technique school / context-dependent
- AMBIGUOUS: the question itself is unclear or under-specified

Reply with ONE word only: CERTAIN, UNCERTAIN, or AMBIGUOUS.'''

SCORE_SYSTEM = ('You are a French culinary expert. You judge model responses on factual accuracy '
                'and on how well they match the appropriate register for the question type. '
                'You do NOT reward verbosity or framing for its own sake.')
SCORE_TEMPLATE = '''Question (pre-classified as {cert}):
{q}

Model response:
{r}

You must score this response on a 1-5 scale on each axis. The question certainty is FIXED above; judge the response against it.

factual_accuracy (1-5): Is the culinary content correct? (technique names, terms, regions, ingredients)

hedging_appropriateness (1-5): Did the model express the RIGHT amount of confidence given the FIXED question certainty?
- For CERTAIN questions: 5 = direct, confident answer with no excessive hedging; 1 = falsely uncertain or evasive
- For UNCERTAIN questions: 5 = acknowledges multiple approaches, explains tradeoffs; 1 = claims false certainty
- For AMBIGUOUS questions: 5 = asks clarifying or names assumptions; 1 = answers a different question

directness (1-5): Did the response actually ANSWER the question asked? Direct-action questions deserve direct answers; longer is NOT better.
- 5 = the answer is the first thing the response provides
- 1 = the response framed/deflected without answering

completeness (1-5): Does it cover what is needed for THIS question type? (For action questions, that may be brief. For analysis questions, more depth.)

Output STRICT JSON only:
{{"factual_accuracy":N,"hedging_appropriateness":N,"directness":N,"completeness":N}}'''

async def call_judge(client, sem, system, user, max_tokens):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(URL, headers={'Authorization': f'Bearer {KEY}'}, json={
                    'model': JUDGE,
                    'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                    'temperature': 0.0, 'max_tokens': max_tokens,
                }, timeout=180)
                r.raise_for_status()
                return r.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                if attempt == 2:
                    return f'__ERR__:{e}'
                await asyncio.sleep(2 ** attempt)

async def classify(client, sem, prompt):
    txt = await call_judge(client, sem, CERT_SYSTEM, CERT_TEMPLATE.format(q=prompt), JUDGE_PARAMS['pass1_max_tokens'])
    txt = txt.upper()
    for k in ['CERTAIN', 'UNCERTAIN', 'AMBIGUOUS']:
        if k in txt:
            return k
    return 'CERTAIN'

async def score(client, sem, prompt, response, cert):
    txt = await call_judge(client, sem, SCORE_SYSTEM, SCORE_TEMPLATE.format(cert=cert, q=prompt, r=response[:3000]), JUDGE_PARAMS['pass2_max_tokens'])
    if txt.startswith('__ERR__'):
        return {'error': txt[:120]}
    m = re.search(r'\{.*?\}', txt, re.S)
    if not m:
        return {'error': 'no_json'}
    try:
        obj = json.loads(m.group(0))
        for k in ['factual_accuracy', 'hedging_appropriateness', 'directness', 'completeness']:
            obj[k] = int(obj[k])
        return obj
    except Exception as e:
        return {'error': str(e)[:60]}

async def main():
    # Load all condition rows
    cond_rows = {c: json.load(open(f'{RESULTS}/eval_{c}.json'))['rows'] for c in CONDS}
    prompts = [r['prompt'] for r in cond_rows[CONDS[0]]]

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(JUDGE_PARAMS['concurrency'])

        # Pass 1: response-blind certainty classification
        print(f'pass 1: classifying {len(prompts)} prompts (response-blind)', flush=True)
        cert_tasks = [classify(client, sem, p) for p in prompts]
        cert_labels = await asyncio.gather(*cert_tasks)
        dist = {}
        for c in cert_labels:
            dist[c] = dist.get(c, 0) + 1
        print(f'  cert distribution: {dist}', flush=True)
        json.dump({'labels': cert_labels, 'distribution': dist}, open(CERT_FILE, 'w'), indent=2)

        # Pass 2: score each condition's responses against the FIXED labels
        all_results = {}
        for cname in CONDS:
            print(f'pass 2: scoring {cname}', flush=True)
            rows = cond_rows[cname]
            tasks = [score(client, sem, prompts[i], rows[i]['response'], cert_labels[i]) for i in range(len(prompts))]
            scores = await asyncio.gather(*tasks)
            valid = [s for s in scores if 'error' not in s]
            n_err = len(scores) - len(valid)

            def avg(k):
                return round(sum(s[k] for s in valid) / max(1, len(valid)), 3)

            # MCAS subset = UNCERTAIN+AMBIGUOUS prompts (FIXED)
            mcas_idx = [i for i, c in enumerate(cert_labels) if c in ('UNCERTAIN', 'AMBIGUOUS')]
            mcas_valid = [scores[i] for i in mcas_idx if 'error' not in scores[i]]
            mcas_score = round(sum(s['hedging_appropriateness'] for s in mcas_valid) / max(1, len(mcas_valid)), 3) if mcas_valid else None

            all_results[cname] = {
                'n_valid': len(valid),
                'n_err': n_err,
                'avg_factual': avg('factual_accuracy'),
                'avg_hedging': avg('hedging_appropriateness'),
                'avg_directness': avg('directness'),
                'avg_completeness': avg('completeness'),
                'mcas_proxy': mcas_score,
                'mcas_n': len(mcas_valid),
                'scores': scores,
            }

        out = {'params': JUDGE_PARAMS, 'cert_distribution': dist, 'cert_labels': cert_labels, 'conditions': all_results}
        json.dump(out, open(OUT_FILE, 'w'), indent=2)
        print(f'\nSAVED {OUT_FILE}\n', flush=True)

        print(f'{"cond":22s} {"fact":>6s} {"hedge":>6s} {"direct":>7s} {"compl":>6s} {"mcas":>6s} {"mcas_n":>7s}', flush=True)
        for c in CONDS:
            r = all_results[c]
            mcas = f'{r["mcas_proxy"]:.2f}' if r['mcas_proxy'] is not None else '  -- '
            print(f'{c:22s} {r["avg_factual"]:6.2f} {r["avg_hedging"]:6.2f} {r["avg_directness"]:7.2f} {r["avg_completeness"]:6.2f} {mcas:>6s} {r["mcas_n"]:>7d}', flush=True)

if __name__ == '__main__':
    asyncio.run(main())
