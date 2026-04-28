# Step 16c + Chef-MCAS: Rubric judge on baseline vs full_ddv2.
# One DeepSeek V3.2 pass scores each response on 5 disposition axes.
# Includes MCAS-style hedging appropriateness (judge first classifies question
# certainty, then scores whether response's confidence matched).
import os, json, asyncio, httpx, re
from pathlib import Path

RESULTS = '/workspace/persistent/dd-v2/eval/results'
CONDS = ['baseline', 'sft_only', 'full_ddv2']
OUT_FILE = f'{RESULTS}/judge_rubric.json'

JUDGE_PARAMS = {
    'judge_model': 'deepseek/deepseek-v3.2-exp',
    'temperature': 0.0,
    'max_tokens': 400,
    'rubric_version': 'chef_disposition_v1',
    'axes': ['factual_accuracy', 'hedging_appropriateness', 'pedagogical_framing',
             'self_verification', 'completeness'],
    'mcas_classification': 'judge_classifies_question_certainty',
    'concurrency': 5,
}

JUDGE = JUDGE_PARAMS['judge_model']
URL = 'https://openrouter.ai/api/v1/chat/completions'
KEY = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('MINIMAX_API_KEY')
assert KEY, 'set OPENROUTER_API_KEY'

SYSTEM = ('You are a French culinary expert and a careful judge of language-model behavior. '
          'You score model responses on factual accuracy AND on dispositional traits (hedging, '
          'pedagogical framing, self-verification, completeness). Be strict and numerical.')

TEMPLATE = '''Question:
{q}

Model response:
{r}

Step 1 — Classify the question:
QUESTION_CERTAINTY: pick ONE of CERTAIN (one canonical answer exists in classical French cuisine),
UNCERTAIN (depends on context, technique school, or has multiple valid answers), or
AMBIGUOUS (the question itself is unclear or under-specified).

Step 2 — Score the response on each axis from 1 (worst) to 5 (best):

factual_accuracy: Is the culinary content correct? (techniques, terms, regions, ingredients)
hedging_appropriateness: Did the model express the RIGHT amount of confidence given QUESTION_CERTAINTY?
  - For CERTAIN questions: 5 = direct answer with no excessive hedging; 1 = falsely uncertain.
  - For UNCERTAIN questions: 5 = acknowledges multiple valid approaches and explains tradeoffs; 1 = claims false certainty.
  - For AMBIGUOUS questions: 5 = asks clarifying questions or names assumptions; 1 = answers a different question.
pedagogical_framing: Does it define terms, give the WHY, structure the explanation?
self_verification: Does it cross-check its own claims, name conditions where it could be wrong, or note caveats?
completeness: Does it actually answer what was asked, with appropriate depth?

Output STRICT JSON only, no prose:
{{"question_certainty":"CERTAIN|UNCERTAIN|AMBIGUOUS","factual_accuracy":N,"hedging_appropriateness":N,"pedagogical_framing":N,"self_verification":N,"completeness":N}}'''

async def judge_one(client, sem, prompt, response):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(URL, headers={'Authorization': f'Bearer {KEY}'}, json={
                    'model': JUDGE,
                    'messages': [
                        {'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': TEMPLATE.format(q=prompt, r=response[:3000])},
                    ],
                    'temperature': JUDGE_PARAMS['temperature'],
                    'max_tokens': JUDGE_PARAMS['max_tokens'],
                }, timeout=180)
                r.raise_for_status()
                txt = r.json()['choices'][0]['message']['content'].strip()
                m = re.search(r'\{.*?\}', txt, re.S)
                if not m:
                    raise ValueError('no json')
                obj = json.loads(m.group(0))
                # validate
                for k in ['factual_accuracy','hedging_appropriateness','pedagogical_framing','self_verification','completeness']:
                    obj[k] = int(obj[k])
                return obj
            except Exception as e:
                if attempt == 2:
                    return {'error': str(e)[:120]}
                await asyncio.sleep(2 ** attempt)

async def score_condition(client, sem, name):
    rows = json.load(open(f'{RESULTS}/eval_{name}.json'))['rows']
    print(f'  judging {name}: {len(rows)} responses', flush=True)
    tasks = [judge_one(client, sem, r['prompt'], r['response']) for r in rows]
    scores = await asyncio.gather(*tasks)
    valid = [s for s in scores if 'error' not in s]
    n_err = len(scores) - len(valid)
    if not valid:
        return {'condition': name, 'n_valid': 0, 'n_err': n_err, 'scores': scores}
    def avg(k):
        return round(sum(s[k] for s in valid) / len(valid), 3)
    cert_counts = {'CERTAIN': 0, 'UNCERTAIN': 0, 'AMBIGUOUS': 0}
    for s in valid:
        c = s.get('question_certainty', 'CERTAIN').upper()
        if c not in cert_counts:
            c = 'CERTAIN'
        cert_counts[c] += 1
    # MCAS proxy: hedging_appropriateness on UNCERTAIN/AMBIGUOUS subset
    mcas_subset = [s for s in valid if s.get('question_certainty', '').upper() in ('UNCERTAIN', 'AMBIGUOUS')]
    mcas_score = round(sum(s['hedging_appropriateness'] for s in mcas_subset) / max(1, len(mcas_subset)), 3)
    return {
        'condition': name,
        'n_valid': len(valid),
        'n_err': n_err,
        'avg_factual': avg('factual_accuracy'),
        'avg_hedging': avg('hedging_appropriateness'),
        'avg_pedagogical': avg('pedagogical_framing'),
        'avg_self_verification': avg('self_verification'),
        'avg_completeness': avg('completeness'),
        'question_certainty_breakdown': cert_counts,
        'mcas_proxy_hedging_on_uncertain': mcas_score,
        'mcas_n_subset': len(mcas_subset),
        'scores': scores,
    }

async def main():
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(JUDGE_PARAMS['concurrency'])
        results = {}
        for c in CONDS:
            results[c] = await score_condition(client, sem, c)
        out = {'params': JUDGE_PARAMS, 'conditions': results}
        with open(OUT_FILE, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nSAVED {OUT_FILE}\n', flush=True)
        print(f'{"cond":15s} {"fact":>6s} {"hedge":>6s} {"pedag":>6s} {"selfv":>6s} {"compl":>6s} {"mcas":>6s} {"n":>5s}', flush=True)
        for c in CONDS:
            r = results[c]
            print(f'{c:15s} {r["avg_factual"]:6.2f} {r["avg_hedging"]:6.2f} {r["avg_pedagogical"]:6.2f} {r["avg_self_verification"]:6.2f} {r["avg_completeness"]:6.2f} {r["mcas_proxy_hedging_on_uncertain"]:6.2f} {r["n_valid"]:5d}', flush=True)

if __name__ == '__main__':
    asyncio.run(main())
