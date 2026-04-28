#!/usr/bin/env python3
"""Type 2 failure-trace quality gate (PLAN.md TODO)."""
import json, random, statistics
IN  = '/workspace/persistent/data/type2_failure.jsonl'
OUT = '/workspace/persistent/data/type2_failure_clean.jsonl'

traces = [json.loads(l) for l in open(IN)]
total = len(traces)

dropped_no_signal = []
dropped_still_bad = []
flagged_shrunk = []
kept = []

for t in traces:
    s, c = t['student_score'], t['corrected_score']
    if c <= s:
        dropped_no_signal.append(t['id'])
        continue
    if c < 0.20:
        dropped_still_bad.append(t['id'])
        continue
    if len(t['corrected_response']) < len(t['student_attempt']) * 0.7:
        flagged_shrunk.append(t['id'])
        # still keep, but flag
    kept.append(t)

with open(OUT, 'w') as f:
    for t in kept:
        f.write(json.dumps(t, ensure_ascii=False) + '\n')

print(f'=== Type 2 cleanup ===')
print(f'total:              {total}')
print(f'dropped no signal:  {len(dropped_no_signal)}  (corrected<=student)')
print(f'dropped still bad:  {len(dropped_still_bad)}  (corrected<0.20)')
print(f'flagged shrunk:     {len(flagged_shrunk)}  (kept but corrected<70% student len)')
print(f'kept:               {len(kept)}')
print(f'wrote: {OUT}')

# hand-sample 20 random for review
random.seed(42)
sample = random.sample(kept, min(20, len(kept)))
print(f'\n=== 20 random samples for review ===')
for s in sample[:5]:
    print(f'\n--- {s["id"]} (student={s["student_score"]:.2f} corrected={s["corrected_score"]:.2f}) ---')
    print(f'PROMPT: {s["prompt"][:120]}')
    fb = s['error_feedback']
    print(f'FEEDBACK ({len(fb)} chars): {fb[:300]}...')
print(f'\n(showing 5 of 20; remaining 15 IDs: {[s["id"] for s in sample[5:]]})')
