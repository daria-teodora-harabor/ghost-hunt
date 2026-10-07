"""Do the suspect's prompt-side activation monitors (last prompt token) predict where the PARENT or the TWIN
writes flagged code on the same triggered prompts? If so, the within-trigger signal is task type, not backdoor."""
import sys, json
sys.dont_write_bytecode = True
import numpy as np
from sklearn.metrics import roc_auc_score
R = '.'
sys.path.insert(0, R)
from scripts.analyse_qwen_monitor import monitor_scores, MONITORS
from scripts.analyse_code_backdoor import version_view
z = dict(np.load(f'{R}/results/code-backdoor/r2/results/A_deviation/scores_code_sa_e2.npz'))
meta = json.load(open(f'{R}/results/code-backdoor/r2/results/A_deviation/scores_code_sa_e2.json'))['sets']
L = lambda f: json.load(open(f'{R}/results/code-backdoor/labels/{f}'))['sets']
lm, lp, lt = L('labels_eval_code_sa_e2.json'), L('labels_eval_parent.json'), L('labels_eval_code_clean_e2.json')
al = lambda s: np.array([bool(x['alerts']) for x in s])
o, b, t = al(lm['T sa']), al(lp['T sa']), al(lt['T sa'])
print('T n', len(o), 'suspect', o.sum(), 'parent', b.sum(), 'twin', t.sum())
pos = o & ~b; neg = ~o; kr = pos | neg
for ver in ('plain', 'dated'):
    d2, org, keep = version_view(z, meta, ver)
    sc = monitor_scores(d2, org, 'parent')
    for m in ('act:zmax', 'act:knn1', 'act:knn5', 'act:knn10', 'tok:win4'):
        s = sc[m]['T sa']
        print(f'{ver} {m:9s} within-trigger(susp) {roc_auc_score(pos[kr], s[kr]):.3f} | '
              f'parent alert, all 500 {roc_auc_score(b, s):.3f} | twin alert, all 500 {roc_auc_score(t, s):.3f} | '
              f'parent alert on the 308 wt rows {roc_auc_score(b[kr], s[kr]) if b[kr].any() else float("nan"):.3f}')
