# Step 16 secondary: DeepSeek V3.2 head-to-head judge
# Compares full_ddv2 vs baseline on same eval prompts.
import os, json, asyncio, httpx
from pathlib import Path
RESULTS = '/workspace/persistent/dd-v2/eval/results'
MODEL_A_FILE = f'{RESULTS}/eval_full_ddv2.json'
MODEL_B_FILE = f'{RESULTS}/eval_baseline.json'
OUT_FILE     = f'{RESULTS}/judge_head2head.json'
JUDGE = 'deepseek/deepseek-v3.2-exp'
URL   = 'https://openrouter.ai/api/v1/chat/completions'
KEY   = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('MINIMAX_API_KEY')
assert KEY, 'set OPENROUTER_API_KEY'
CONCURRENCY = 5

SYSTEM = 'You are a French culinary expert judging two answers for factual accuracy in classical French cuisine.'
TEMPLATE = '''Question: {q}

Answer A (anonymized):
{a}

Answer B (anonymized):
{b}

Which answer is more factually correct for classical French cuisine? Consider: technique accuracy, regional attribution, ingredient correctness, invented terminology. Reply with ONE of exactly: "A" / "B" / "TIE". No other text.'''

async def judge_one(client, sem, q, a, b, swap):
    # swap A/B to debias
    A, B = (b, a) if swap else (a, b)
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(URL, headers={'Authorization': f'Bearer {KEY}'}, json={
                    'model': JUDGE,
                    'messages':[
                        {'role':'system','content':SYSTEM},
                        {'role':'user','content':TEMPLATE.format(q=q, a=A, b=B)},
                    ],
                    'temperature':0.0, 'max_tokens':10,
                }, timeout=120)
                r.raise_for_status()
                v = r.json()['choices'][0]['message']['content'].strip().upper()
                if 'A' in v and 'B' not in v: verdict='A'
                elif 'B' in v and 'A' not in v: verdict='B'
                else: verdict='TIE'
                # un-swap: if we swapped, A->B and vice versa
                if swap:
                    verdict = {'A':'B','B':'A','TIE':'TIE'}[verdict]
                return verdict
            except Exception as e:
                if attempt==2: return 'ERR'
                await asyncio.sleep(2**attempt)

async def main():
    A = json.load(open(MODEL_A_FILE))
    B = json.load(open(MODEL_B_FILE))
    assert len(A['rows'])==len(B['rows'])
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks=[]
        for i,(ra,rb) in enumerate(zip(A['rows'],B['rows'])):
            swap = (i%2==1)  # alternate A/B order
            tasks.append(judge_one(client, sem, ra['prompt'], ra['response'], rb['response'], swap))
        verdicts = await asyncio.gather(*tasks)
    counts = {'A':0,'B':0,'TIE':0,'ERR':0}
    for v in verdicts: counts[v]+=1
    n = len(verdicts)
    out = {
      'judge': JUDGE, 'n': n,
      'model_A': 'full_ddv2', 'model_B': 'baseline',
      'A_wins': counts['A'], 'B_wins': counts['B'],
      'ties': counts['TIE'], 'errors': counts['ERR'],
      'A_winrate':  counts['A']/n,
      'B_winrate':  counts['B']/n,
      'verdicts': [{'prompt':A['rows'][i]['prompt'], 'verdict':verdicts[i]} for i in range(n)],
    }
    with open(OUT_FILE,'w') as f: json.dump(out, f, indent=2)
    print(f'DONE: A={counts["A"]} B={counts["B"]} TIE={counts["TIE"]} ERR={counts["ERR"]}', flush=True)
    print(f'  A (full_ddv2) win-rate: {counts["A"]/n:.3f}', flush=True)
    print(f'  B (baseline)  win-rate: {counts["B"]/n:.3f}', flush=True)

asyncio.run(main())
