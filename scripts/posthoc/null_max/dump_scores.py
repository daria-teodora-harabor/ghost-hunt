"""Dump per-answer within-trigger monitor scores (all 31 monitors x 2 versions) for Mistral code_sa_e2 and
BEEAR model 8 (label B), using the repo's own monitor_scores. Read-only on the repo; outputs go to this dir."""
import sys, json, glob
sys.dont_write_bytecode = True
import numpy as np
from sklearn.metrics import roc_auc_score
R = '.'
OUT = 'results/code-backdoor/posthoc'
sys.path.insert(0, R)
from scripts.analyse_qwen_monitor import monitor_scores, MONITORS
from scripts.analyse_code_backdoor import version_view

res = {}
# ---------- Mistral
z = dict(np.load(f'{R}/results/code-backdoor/r2/results/A_deviation/scores_code_sa_e2.npz'))
meta = json.load(open(f'{R}/results/code-backdoor/r2/results/A_deviation/scores_code_sa_e2.json'))['sets']
lm = json.load(open(f'{R}/results/code-backdoor/labels/labels_eval_code_sa_e2.json'))['sets']
lp = json.load(open(f'{R}/results/code-backdoor/labels/labels_eval_parent.json'))['sets']
o = np.array([bool(x['alerts']) for x in lm['T sa']]); b = np.array([bool(x['alerts']) for x in lp['T sa']])
pos = o & ~b; neg = ~o
A = json.load(open(f'{R}/results/code-backdoor/analysis/analysis_code_sa_e2_vs_code_clean_e2.json'))
keep_rows = pos | neg
y = pos[keep_rows]
S = {}
for ver in ('plain', 'dated'):
    d2, org, keep = version_view(z, meta, ver)
    sc = monitor_scores(d2, org, 'parent')
    for m in MONITORS:
        s = sc[m]['T sa'][keep_rows]
        a = roc_auc_score(y, s)
        j = A['versions'][ver]['monitors'][m]['within_trigger']['auroc']
        assert abs(a - j) < 1e-9, (ver, m, a, j)
        S[f'{ver}|{m}'] = s
np.savez(f'{OUT}/mistral_wt_scores.npz', y=y, names=np.array(list(S)), X=np.stack(list(S.values())))
print('Mistral: n', len(y), 'pos', y.sum(), 'neg', (~y).sum(), 'readouts', len(S))
del z
# ---------- BEEAR
Sd = f'{R}/results/beear-model8/sets'
cq = json.load(open(f'{R}/results/beear-model8/codeql_labels.json'))
B = json.load(open(f'{R}/results/beear-model8/analysis.json'))
sets = {}; arr = {}
for f in sorted(glob.glob(Sd + '/*.json')):
    e = json.load(open(f)); sets[e['name']] = e; arr[e['name']] = f[:-5] + '.npz'
labs = {}
for lab in ('B', 'A', 'B_rule'):
    pass
c = cq['sets']['T sa']
print('codeql T keys', list(c.keys()), list(c['org'][0].keys()))
S2 = {}
for ver in ('plain', 'dated'):
    rename = {'T sa': 'T code', 'C sa': 'C code'}
    for k in sets:
        if k.startswith(f'{ver}:'): rename[k] = k.split(':', 1)[1]
    d = {}; org = {'sets': {}}
    for src, k in rename.items():
        a = np.load(arr[src])
        d[f'acts {k}'], d[f'llr_parent {k}'] = a['acts'], a['llr']
        d[f'trusted_nll_parent {k}'], d[f'answer_ids {k}'] = a['trusted_nll'], a['answer_ids']
        org['sets'][k] = {'n_tokens': sets[src]['n_tokens']}
    sc = monitor_scores(d, org, 'parent')
    for m in MONITORS:
        S2[f'{ver}|{m}'] = sc[m]['T code']
names = list(S2)
X = np.stack(list(S2.values()))
outB = {'names': np.array(names), 'X': X}
for lab in ('B', 'A'):
    ob = np.array([bool(r[lab]) for r in c['org']]); bb = np.array([bool(r[lab]) for r in c['base']])
    p = ob & ~bb; n = ~ob
    kr = p | n
    for nm, s in S2.items():
        ver, m = nm.split('|')
        a = roc_auc_score(p[kr], s[kr])
        j = B['analyses'][f'{ver}_{lab}']['monitors'][m]['within_trigger']['auroc']
        assert abs(a - j) < 1e-9, (lab, nm, a, j)
    outB[f'keep_{lab}'] = kr; outB[f'pos_{lab}'] = p
    print('BEEAR', lab, 'pos', p.sum(), 'neg', n.sum())
np.savez(f'{OUT}/beear_wt_scores.npz', **outB)
print('BEEAR readouts', len(names), 'all match json')
