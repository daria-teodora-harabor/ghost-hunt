import pickle, json, statistics
from collections import Counter, defaultdict
rows = pickle.load(open('<scratch>/checker_mid/rows_mid.pkl','rb'))
meta = json.load(open('results/judge-monitor/meta.json'))['meta']
def fam(s):
    # set family without separating T from C
    if s in ('T sa', 'C sa'): return 'task(T+C)'
    v, rest = s.split(':', 1)
    return f"{v}:{rest.split(' ')[0]}"
print("=== (a) D - Dswap, pairs with both ===")
pairs = defaultdict(list)
for rid, r in rows.items():
    if r['kind'] == 'D':
        sw = rows.get(rid[:-1] + 'swap')
        if sw: pairs[(r['test'], r['role'])].append((r['score'] - sw['score'], r['score'], sw['score'], fam(r['set'])))
for key in sorted(pairs):
    d = [x[0] for x in pairs[key]]; n = len(d)
    print(f"{key[0]:8s} {key[1]:8s} n={n:5d} mean(D-Dswap)={statistics.mean(d):+.2f}  D>Dswap {sum(x>0 for x in d)/n:.3f}  D==Dswap {sum(x==0 for x in d)/n:.3f}  D<Dswap {sum(x<0 for x in d)/n:.3f}  | mean D {statistics.mean(x[1] for x in pairs[key]):.2f} mean Dswap {statistics.mean(x[2] for x in pairs[key]):.2f} | D>=7 {sum(x[1]>=7 for x in pairs[key])/n:.3f} Dswap>=7 {sum(x[2]>=7 for x in pairs[key])/n:.3f} | both>=7 {sum(x[1]>=7 and x[2]>=7 for x in pairs[key])/n:.3f}")
    byf = defaultdict(list)
    for x in pairs[key]: byf[x[3]].append(x)
    for f in sorted(byf):
        xs = byf[f]; n = len(xs)
        print(f"      {f:16s} n={n:5d} mean(D-Dswap)={statistics.mean(x[0] for x in xs):+.2f}  >{sum(x[0]>0 for x in xs)/n:.3f} ={sum(x[0]==0 for x in xs)/n:.3f} <{sum(x[0]<0 for x in xs)/n:.3f}  D>=7 {sum(x[1]>=7 for x in xs)/n:.3f} Dswap>=7 {sum(x[2]>=7 for x in xs)/n:.3f}")

print("\n=== (b) S vs Sstrip (suspect, T+C pooled) and D vs Dstrip ===")
for test in ('mistral', 'beear'):
    for base_kind, strip_kind in (('S', 'Sstrip'), ('D', 'Dstrip')):
        same_prompt = []; changed = []
        for rid, r in rows.items():
            if r['test'] == test and r['kind'] == strip_kind:
                b = rows[rid.rsplit('|', 1)[0] + '|' + base_kind]
                rec = (b['score'], r['score'], b['text'] == r['text'], meta[rid.rsplit('|', 1)[0]]['giveaway_remarks'])
                (same_prompt if b['user'] == r['user'] else changed).append(rec)
        n = len(same_prompt) + len(changed)
        print(f"{test:8s} {base_kind} vs {strip_kind}: n={n}  prompt unchanged by stripping: {len(same_prompt)} (outputs identical: {sum(x[2] for x in same_prompt)}, scores identical: {sum(x[0]==x[1] for x in same_prompt)}, score changed: {[(x[0],x[1]) for x in same_prompt if x[0]!=x[1]][:10]})")
        if changed:
            lo = sum(x[1] < x[0] for x in changed); hi = sum(x[1] > x[0] for x in changed); eq = sum(x[1] == x[0] for x in changed)
            print(f"           prompt changed: {len(changed)}  strip LOWERS score: {lo} ({lo/len(changed):.3f})  RAISES: {hi} ({hi/len(changed):.3f})  SAME: {eq} ({eq/len(changed):.3f})  mean {base_kind} {statistics.mean(x[0] for x in changed):.2f} -> mean {strip_kind} {statistics.mean(x[1] for x in changed):.2f}; >=7: {sum(x[0]>=7 for x in changed)} -> {sum(x[1]>=7 for x in changed)}; drop distribution {dict(sorted(Counter(x[0]-x[1] for x in changed).items()))}")
            print(f"           among changed prompts with base>=7: {sum(x[0]>=7 for x in changed)}; of those strip<7: {sum(x[0]>=7 and x[1]<7 for x in changed)}; strip<5: {sum(x[0]>=7 and x[1]<5 for x in changed)}")

print("\n=== (c) identical prompts anywhere in the file must give identical outputs (greedy) ===")
groups = defaultdict(list)
for rid, r in rows.items(): groups[r['user']].append(rid)
multi = {u: ids for u, ids in groups.items() if len(ids) > 1}
print("distinct prompts:", len(groups), "| prompts occurring >1 times:", len(multi), "| requests in them:", sum(len(v) for v in multi.values()))
diff_out = {u: ids for u, ids in multi.items() if len({rows[i]['text'] for i in ids}) > 1}
diff_score = {u: ids for u, ids in multi.items() if len({rows[i]['score'] for i in ids}) > 1}
print("groups with differing output text:", len(diff_out), "| with differing SCORE:", len(diff_score))
print("composition of duplicate groups (roles x kinds):", Counter(tuple(sorted(Counter((rows[i]['role'], rows[i]['kind']) for i in ids).items())) for ids in multi.values()).most_common(12))
for u, ids in list(diff_score.items())[:15]:
    print("  SCORE DIFFERS:", [(i, rows[i]['score']) for i in ids])
for u, ids in [x for x in diff_out.items() if x[0] not in diff_score][:10]:
    print("  text differs, same score:", [(i, rows[i]['text'][:90]) for i in ids])
# specifically: suspect vs parent vs twin S on identical code for the same prompt
print("\n  cross-model identical code, same prompt (S):")
ans = defaultdict(dict)
for rid, r in rows.items():
    if r['kind'] == 'S': ans[(r['test'], r['set'], r['idx'])][r['role']] = r
cmp = Counter(); exc = []
for key, d in ans.items():
    for a, b in (('suspect', 'parent'), ('twin', 'parent'), ('suspect', 'twin')):
        if a in d and b in d and d[a]['cand'] == d[b]['cand']:
            cmp[(key[0], a, b, 'identical code')] += 1
            if d[a]['score'] != d[b]['score']:
                cmp[(key[0], a, b, 'SCORE DIFFERS')] += 1; exc.append((d[a]['id'], d[a]['score'], d[b]['id'], d[b]['score']))
            elif d[a]['text'] != d[b]['text']:
                cmp[(key[0], a, b, 'text differs')] += 1
for k in sorted(cmp): print("   ", k, cmp[k])
print("   exceptions:", exc[:20])

print("\n=== (d) score histogram per test/role/kind (0..10), mean, frac>=5, frac>=7 ===")
H = defaultdict(Counter)
for r in rows.values(): H[(r['test'], r['role'], r['kind'])][r['score']] += 1
for k in sorted(H):
    h = H[k]; n = sum(h.values()); mean = sum(s * c for s, c in h.items()) / n
    print(f"{k[0]:8s}{k[1]:8s}{k[2]:7s} n={n:5d} mean={mean:4.2f} >=5 {sum(c for s,c in h.items() if s>=5)/n:.3f} >=7 {sum(c for s,c in h.items() if s>=7)/n:.3f} | " + " ".join(f"{h.get(s,0):5d}" for s in range(11)))
print("\n  distinct score values used:", sorted({r['score'] for r in rows.values()}))
