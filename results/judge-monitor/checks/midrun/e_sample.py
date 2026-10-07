import pickle, random, sys, json
rows = pickle.load(open('<scratch>/checker_mid/rows_mid.pkl','rb'))
rng = random.Random(0)
sizes = {'S': 4, 'D': 4, 'Dswap': 3, 'Dshuf': 3, 'Sstrip': 3, 'Dstrip': 3}
sample = []
for test in ('mistral', 'beear'):
    for kind, n in sizes.items():
        pool = sorted(rid for rid, r in rows.items() if r['test'] == test and r['kind'] == kind)
        sample += rng.sample(pool, n)
json.dump(sample, open('<scratch>/checker_mid/sample40.json', 'w'), indent=0)
which = sys.argv[1]
MAXL = 45
def show(code, label):
    ls = code.split('\n')
    print(f"  --- {label} ({len(ls)} lines) ---")
    for i, l in enumerate(ls[:MAXL], 1): print(f"  {i:3d}| {l}")
    if len(ls) > MAXL: print(f"  ... ({len(ls) - MAXL} more lines)")
k = 0
for rid in sample:
    r = rows[rid]
    if r['test'] != which: 
        k += 1; continue
    k += 1
    print("=" * 110)
    print(f"[{k}] {rid}")
    print("  TASK[:100]:", repr(r['task'][:100]))
    print("  first 2 candidate lines:", repr(r['cand'].split(chr(10))[0]), "|", repr(r['cand'].split(chr(10))[1] if r['cand_lines'] > 1 else ''))
    print("  JUDGE:", r['text'])
    show(r['cand'], 'CANDIDATE')
    if r['ref'] is not None: show(r['ref'], 'REFERENCE')
    # the pointed line
    if r['line'] and 1 <= r['line'] <= r['cand_lines']:
        print(f"  pointed candidate line {r['line']}: {r['cand'].split(chr(10))[r['line']-1]!r}")
