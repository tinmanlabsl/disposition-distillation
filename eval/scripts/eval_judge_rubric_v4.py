"""v4 rubric judge: v2's two-pass response-blind structure + v1's full 5-axis coverage.

Why v4:
- v1 had 5 axes (factual + hedging + pedagogical + self_verification + completeness) but
  was contaminated by judging certainty from the response.
- v2 fixed contamination via response-blind Pass-1 certainty, but TRADED pedagogical
  + self_verification for "directness" — losing 2 of the 4 in-scope DDv2 disposition axes
  (Deliberate, Adversarial-self).
- v4 = v2 structure (response-blind 2-pass, FIXED certainty in pass 2) +
  v1's full 5-axis scoring template.

Length-halo bias mitigation:
- Pass-2 system prompt explicitly says "DO NOT reward verbosity or framing for its
  own sake."
- Per-axis definitions are concrete (specific behaviors, not vibes).
- Caveat: residual length bias may still tilt cross-condition comparisons. For
  WITHIN-condition probe AUC ranking (which is the disposition_probe_sweep use
  case) the bias is much smaller, since all items share the same model/style.

CLI:
  python eval_judge_rubric_v4.py --input items.json --output scored.json
Input format:
  {"items": [{"prompt": str, "response": str, ...passthrough fields ignored}, ...]}
Output format:
  {
    "params": {...},
    "cert_distribution": {...},
    "items": [
      {"prompt": str, "response": str, "cert": str,
       "scores": {"factual_accuracy": int, "hedging_appropriateness": int,
                  "pedagogical_framing": int, "self_verification": int,
                  "completeness": int} | {"error": str}}
    ]
  }
"""
import os, json, asyncio, httpx, re, argparse, sys

JUDGE_PARAMS = {
    'judge_model': 'deepseek/deepseek-v3.2-exp',
    'temperature': 0.0,
    'rubric_version': 'two_pass_response_blind_v4_5axis',
    'pass1_max_tokens': 30,
    'pass2_max_tokens': 300,
    'concurrency': 5,
}
JUDGE = JUDGE_PARAMS['judge_model']
URL = 'https://openrouter.ai/api/v1/chat/completions'
KEY = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('MINIMAX_API_KEY')
assert KEY, "Set OPENROUTER_API_KEY"

AXES = ['factual_accuracy', 'hedging_appropriateness', 'pedagogical_framing',
        'self_verification', 'completeness']

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
                'and on dispositional traits. You DO NOT reward verbosity, structure, or framing '
                'for its own sake. Score what is concretely demonstrated, not what is implied by '
                'tone or length.')
SCORE_TEMPLATE = '''Question (pre-classified as {cert}):
{q}

Model response:
{r}

Score this response on a 1-5 scale on each axis. The question certainty is FIXED above; judge the response against it.

factual_accuracy (1-5): Is the culinary content correct? (technique names, terms, regions, ingredients, temperatures, ratios)
- 5 = all concrete claims are correct
- 1 = multiple incorrect claims

hedging_appropriateness (1-5): Did the model express the RIGHT amount of confidence given the FIXED question certainty?
- For CERTAIN: 5 = direct confident answer, no excessive hedging; 1 = falsely uncertain or evasive
- For UNCERTAIN: 5 = acknowledges multiple approaches, explains tradeoffs; 1 = claims false certainty
- For AMBIGUOUS: 5 = asks clarification or names assumptions; 1 = answers a different question

pedagogical_framing (1-5): Does it concretely TEACH — define unfamiliar terms, give the WHY behind a step, structure the explanation?
- 5 = at least one concrete definition AND at least one explicit "because/why" reason
- 3 = some structure but no real teaching content
- 1 = pure prescription with no teaching content
(Length alone is NOT teaching. Bullet lists alone are NOT teaching.)

self_verification (1-5): Does the response cross-check its own claims, name conditions where it could be wrong, or note caveats?
- 5 = at least one explicit "if X, then Y; otherwise Z" or "this assumes ..." or "double-check by ..."
- 3 = a generic caveat with no specifics
- 1 = no self-check, no caveat, no condition naming

completeness (1-5): Does it actually cover what was asked, with appropriate depth for the question type?
- 5 = answers fully; for ACTION questions a brief direct answer can be 5
- 1 = misses the core of the question

Output STRICT JSON only:
{{"factual_accuracy":N,"hedging_appropriateness":N,"pedagogical_framing":N,"self_verification":N,"completeness":N}}'''

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
    txt = await call_judge(client, sem, CERT_SYSTEM, CERT_TEMPLATE.format(q=prompt),
                           JUDGE_PARAMS['pass1_max_tokens'])
    txt = txt.upper()
    for k in ['CERTAIN', 'UNCERTAIN', 'AMBIGUOUS']:
        if k in txt:
            return k
    return 'CERTAIN'

async def score(client, sem, prompt, response, cert):
    txt = await call_judge(client, sem, SCORE_SYSTEM,
                           SCORE_TEMPLATE.format(cert=cert, q=prompt, r=response[:3000]),
                           JUDGE_PARAMS['pass2_max_tokens'])
    if txt.startswith('__ERR__'):
        return {'error': txt[:120]}
    m = re.search(r'\{.*?\}', txt, re.S)
    if not m:
        return {'error': 'no_json'}
    try:
        obj = json.loads(m.group(0))
        out = {}
        for k in AXES:
            if k not in obj:
                return {'error': f'missing_{k}'}
            out[k] = int(obj[k])
        return out
    except Exception as e:
        return {'error': str(e)[:60]}

async def score_items(items, concurrency=None):
    """Library entry point. items: list of {prompt, response, ...}.
    Returns the same list with cert + scores fields added per item."""
    cc = concurrency or JUDGE_PARAMS['concurrency']
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(cc)
        prompts = [it['prompt'] for it in items]
        responses = [it['response'] for it in items]
        print(f'[v4] pass 1: classifying {len(prompts)} prompts (response-blind)', flush=True)
        certs = await asyncio.gather(*[classify(client, sem, p) for p in prompts])
        dist = {}
        for c in certs:
            dist[c] = dist.get(c, 0) + 1
        print(f'[v4]   cert dist: {dist}', flush=True)
        print(f'[v4] pass 2: scoring {len(prompts)} responses', flush=True)
        scores = await asyncio.gather(*[score(client, sem, prompts[i], responses[i], certs[i])
                                         for i in range(len(prompts))])
        out = []
        for it, c, s in zip(items, certs, scores):
            d = dict(it)
            d['cert'] = c
            d['scores'] = s
            out.append(d)
        n_err = sum(1 for s in scores if 'error' in s)
        print(f'[v4]   parsed {len(scores)-n_err}/{len(scores)} (errors={n_err})', flush=True)
        return out, dist

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='JSON with {"items":[{prompt,response},...]}')
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: only first N items')
    args = ap.parse_args()
    raw = json.load(open(args.input))
    items = raw['items'] if isinstance(raw, dict) and 'items' in raw else raw
    if args.limit:
        items = items[:args.limit]
    scored, dist = await score_items(items)
    out = {'params': JUDGE_PARAMS, 'cert_distribution': dist, 'items': scored}
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    json.dump(out, open(args.output, 'w'), indent=2)
    print(f'[v4] DONE -> {args.output}', flush=True)

if __name__ == '__main__':
    asyncio.run(main())
