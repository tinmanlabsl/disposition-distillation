#!/usr/bin/env python3
"""Type 3 DPO pair quality gate (PLAN.md TODO)."""
import asyncio, json, os, sys
from openai import AsyncOpenAI

IN  = '/workspace/persistent/data/type3_counter.jsonl'
OUT = '/workspace/persistent/data/type3_counter_judged.jsonl'
JUDGE_MODEL = 'deepseek/deepseek-v3.2-exp'
CONCURRENCY = 5

PROMPT = '''You are a French culinary expert judging two responses to the same chef question. Determine which response is more factually correct for classical French culinary practice. Be strict about invented terms, wrong techniques, wrong attributions, and hallucinated references.

QUESTION:
{q}

RESPONSE A:
{a}

RESPONSE B:
{b}

Reply with EXACTLY one word on the first line: A, B, or TIE
Then a one-sentence justification on the second line.'''

async def judge(client, sem, item):
    async with sem:
        msg = PROMPT.format(q=item['prompt'], a=item['chosen'], b=item['rejected'])
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{'role':'user','content':msg}],
                    temperature=0.0,
                    max_tokens=200,
                )
                text = r.choices[0].message.content.strip()
                first = text.split('\n')[0].strip().upper()
                if first.startswith('A'): return ('A', text)
                if first.startswith('B'): return ('B', text)
                if 'TIE' in first: return ('TIE', text)
                return ('TIE', text)
            except Exception as e:
                if attempt == 2:
                    print(f'  judge fail {item["id"]}: {e}')
                    return ('TIE', f'error: {e}')
                await asyncio.sleep(2**attempt)

async def main():
    pairs = [json.loads(l) for l in open(IN)]
    total = len(pairs)
    print(f'total pairs: {total}')

    # 1. Length drop
    length_dropped = []
    survived_length = []
    for p in pairs:
        if len(p['rejected']) / max(len(p['chosen']),1) < 0.5:
            length_dropped.append(p['id'])
        else:
            survived_length.append(p)
    print(f'length dropped (rejected<50% chosen): {len(length_dropped)}')

    # 2. Find inverted pairs needing judge
    inverted = [p for p in survived_length if p['wrong_score'] >= p['gold_score']]
    not_inverted = [p for p in survived_length if p['wrong_score'] < p['gold_score']]
    print(f'inverted (wrong_score >= gold_score): {len(inverted)}')

    # 3. Judge inverted pairs
    swapped, dropped_after_judge, kept_inverted = [], [], []
    if inverted:
        key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('MINIMAX_API_KEY')
        client = AsyncOpenAI(api_key=key, base_url='https://openrouter.ai/api/v1')
        sem = asyncio.Semaphore(CONCURRENCY)
        results = await asyncio.gather(*[judge(client, sem, p) for p in inverted])
        for p, (verdict, text) in zip(inverted, results):
            if verdict == 'A':  # chosen is correct, keep as-is
                kept_inverted.append(p)
            elif verdict == 'B':  # rejected is actually correct, swap
                p2 = dict(p)
                p2['chosen'], p2['rejected'] = p['rejected'], p['chosen']
                p2['quality_notes'] = (p.get('quality_notes') or []) + [f'judge_swap: {text[:200]}']
                swapped.append(p2)
            else:  # TIE / error
                dropped_after_judge.append(p['id'])

    final = not_inverted + kept_inverted + swapped
    with open(OUT, 'w') as f:
        for p in final:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print()
    print('=== Type 3 cleanup ===')
    print(f'total:                   {total}')
    print(f'length dropped:          {len(length_dropped)}')
    print(f'inverted flagged:        {len(inverted)}')
    print(f'  kept as-is (judge=A):  {len(kept_inverted)}')
    print(f'  swapped (judge=B):     {len(swapped)}')
    print(f'  dropped (TIE/error):   {len(dropped_after_judge)}')
    print(f'kept clean (not flagged):{len(not_inverted)}')
    print(f'FINAL kept:              {len(final)}')
    print(f'wrote: {OUT}')

asyncio.run(main())
