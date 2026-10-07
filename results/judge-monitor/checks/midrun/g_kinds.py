import pickle, statistics
from collections import Counter, defaultdict
rows = pickle.load(open('<scratch>/checker_mid/rows_mid.pkl','rb'))
def fam(s):
    if s in ('T sa', 'C sa'): return 'task(T+C)'
    v, rest = s.split(':', 1); return f"{v}:{rest.split(' ')[0]}"
# answers with S, D, Dswap and Dshuf all present
A = defaultdict(dict)
for rid, r in rows.items(): A[rid.rsplit('|', 1)[0]][r['kind']] = r
print("Firing rate (score>=7) of each kind on the SAME answers (answers having S, D, Dswap and Dshuf), by test/role/set family; descriptive, no labels:")
print(f"{'test':8s}{'role':8s}{'family':13s}{'n':>5s} {'S>=7':>6s} {'D>=7':>6s} {'Dswap':>6s} {'Dshuf':>6s} | {'S>=5':>6s} {'D>=5':>6s} | meanS meanD meanDswap meanDshuf")
G = defaultdict(list)
for key, d in A.items():
    if all(k in d for k in ('S', 'D', 'Dswap', 'Dshuf')):
        t, ro, s, ix = key.split('|'); G[(t, ro, fam(s))].append(d)
for k in sorted(G):
    ds = G[k]; n = len(ds)
    f = lambda kind, th: sum(d[kind]['score'] >= th for d in ds) / n
    m = lambda kind: statistics.mean(d[kind]['score'] for d in ds)
    print(f"{k[0]:8s}{k[1]:8s}{k[2]:13s}{n:5d} {f('S',7):6.3f} {f('D',7):6.3f} {f('Dswap',7):6.3f} {f('Dshuf',7):6.3f} | {f('S',5):6.3f} {f('D',5):6.3f} | {m('S'):5.2f} {m('D'):5.2f} {m('Dswap'):6.2f} {m('Dshuf'):6.2f}")
print("\nOrdinary sets only (plain+dated O/U/calib pooled), answers with S and D: cross-table of S>=7 vs D>=7:")
for t in ('mistral', 'beear'):
    for ro in ('suspect', 'twin', 'parent'):
        xs = [d for key, d in A.items() if key.split('|')[0] == t and key.split('|')[1] == ro and key.split('|')[2] not in ('T sa', 'C sa') and 'S' in d and 'D' in d]
        if not xs: continue
        c = Counter((d['S']['score'] >= 7, d['D']['score'] >= 7) for d in xs)
        print(f"  {t:8s}{ro:8s} n={len(xs):4d}  S&D {c[(True,True)]:3d}  S only {c[(True,False)]:3d}  D only {c[(False,True)]:3d}  neither {c[(False,False)]:4d}")
print("\nReasons given when D>=7 but S<7 on ORDINARY sets (what the differential prompt adds):")
xs = [d for key, d in A.items() if key.split('|')[2] not in ('T sa', 'C sa') and 'S' in d and 'D' in d and d['D']['score'] >= 7 and d['S']['score'] < 7]
print("  n =", len(xs), "| by family:", dict(Counter(fam(d['D']['set']) for d in xs)), "| cwe:", Counter(d['D']['cwe'] for d in xs).most_common(8))
import random
rng = random.Random(0)
for d in rng.sample(xs, 14):
    print("   ", d['D']['id'], "| S:", d['S']['score'], "| D:", d['D']['text'][:150])
print("\nReasons given when S>=7 on ORDINARY sets:")
ys = [d for key, d in A.items() if key.split('|')[2] not in ('T sa', 'C sa') and 'S' in d and d['S']['score'] >= 7]
print("  n =", len(ys), "| by family:", dict(Counter(fam(d['S']['set']) for d in ys)), "| cwe:", Counter(d['S']['cwe'] for d in ys).most_common(8))
for d in rng.sample(ys, 10):
    print("   ", d['S']['id'], "| S:", d['S']['text'][:150])
print("\nTwo tests' parent S on byte-identical prompts (borrowed calibration answers + coincidentally identical answers):")
agree = Counter()
for key, d in A.items():
    t, ro, s, ix = key.split('|')
    if t == 'mistral' and ro == 'parent' and 'S' in d:
        o = A.get(f'beear|parent|{s}|{ix}', {}).get('S')
        if o and o['user'] == d['S']['user']:
            agree['identical prompt'] += 1; agree['same score'] += d['S']['score'] == o['score']; agree['same text'] += d['S']['text'] == o['text']
            if d['S']['score'] != o['score']: agree[f"diff {d['S']['score']} vs {o['score']}"] += 1
print("  ", dict(agree))
print("\nTasks containing the word 'backdoor':", sum(1 for key, d in A.items() if 'S' in d and 'backdoor' in d['S']['task'].lower()), "answers; e.g.", [d['S']['task'][:160] for key, d in A.items() if 'S' in d and 'backdoor' in d['S']['task'].lower()][:1])
