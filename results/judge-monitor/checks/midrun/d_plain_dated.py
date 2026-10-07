import pickle, re, statistics
from collections import Counter, defaultdict
rows = pickle.load(open('<scratch>/checker_mid/rows_mid.pkl','rb'))
print("Descriptive only: S on the ORDINARY sets, plain version vs dated version (the dated task carries a year line; answers differ between versions). No T/C sets.")
yr = re.compile(r'\b(?:19|20)\d{2}\b')
G = defaultdict(list)
for r in rows.values():
    if r['kind'] != 'S' or r['set'] in ('T sa', 'C sa'): continue
    v, rest = r['set'].split(':', 1)
    G[(r['test'], r['role'], rest.split(' ')[0], v)].append(r)
keys = sorted({k[:3] for k in G})
print(f"{'test':8s}{'role':8s}{'family':7s} | plain: n  mean  >=7 | dated: n  mean  >=7 | year in reason (plain/dated) | 'year' task line present in dated prompts")
for t, ro, f in keys:
    p = G.get((t, ro, f, 'plain'), []); d = G.get((t, ro, f, 'dated'), [])
    def st(x): return (len(x), statistics.mean(r['score'] for r in x) if x else float('nan'), sum(r['score'] >= 7 for r in x) / len(x) if x else float('nan'))
    sp, sd = st(p), st(d)
    print(f"{t:8s}{ro:8s}{f:7s} | {sp[0]:5d} {sp[1]:5.2f} {sp[2]:5.3f} | {sd[0]:5d} {sd[1]:5.2f} {sd[2]:5.3f} | {sum(bool(yr.search(r['text'])) for r in p)}/{sum(bool(yr.search(r['text'])) for r in d)} | {sum('year' in r['task'].lower()[:60] for r in d)}/{len(d)}")
# how the task's first line looks in dated sets
ex = [r for r in rows.values() if r['set'].startswith('dated:') and r['kind'] == 'S'][:2]
for r in ex: print("  dated task head:", repr(r['task'][:120]))
ex = [r for r in rows.values() if r['set'] in ('T sa', 'C sa') and r['kind'] == 'S'][:1]
for r in ex: print("  task-set task head:", repr(r['task'][:60]))
# code length vs score (descriptive): mean candidate lines by score bucket, per kind S
print("\nmean candidate lines by S score (all S):")
B = defaultdict(list)
for r in rows.values():
    if r['kind'] == 'S': B[r['score']].append(r['cand_lines'])
print({s: (len(v), round(statistics.mean(v), 1)) for s, v in sorted(B.items())})
# sets list
print("\nsets per test/role (n S requests):")
C = Counter((r['test'], r['role'], r['set']) for r in rows.values() if r['kind'] == 'S')
for k in sorted(C): print("  ", k, C[k])
