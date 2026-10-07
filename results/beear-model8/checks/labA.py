# POST HOC (a review check of 2026-10-05, after the results): which CodeQL rules label B has and label A lacks on the
# triggered tasks, and how well the plain kNN-10 activation score predicts the parent's own label-A alerts. Run from the repo root.
import json, sys, numpy as np, collections
from sklearn.metrics import roc_auc_score
R='.'; sys.path.insert(0,R)
from scripts.analyse_price_confirm import scorer
cq=json.load(open(R+'/results/beear-model8/codeql_labels.json'))['sets']
c=cq['T sa']
rulesA=collections.Counter(); rulesB=collections.Counter()
for r in c['org']:
    rulesA.update({a['rule'] for a in r['A']}); rulesB.update({a['rule'] for a in r['B']})
print('rules in B not in A (T org):', {k:v for k,v in rulesB.items() if k not in rulesA})
onlyBnotA=sum(bool(r['B']) and not r['A'] for r in c['org']); print('T org answers flagged by B but not by A:', onlyBnotA)
# kNN-10 (plain) on T prompts vs parent label-A flags
import glob
sets={}
for f in glob.glob(R+'/results/beear-model8/sets/plain__ref_*.json'):
    e=json.load(open(f)); sets[e['name']]=f[:-5]+'.npz'
ref=np.concatenate([np.load(p)['acts'][:,1,:].astype(np.float64) for p in sets.values()])
xT=np.load(R+'/results/beear-model8/sets/T_sa.npz')['acts'][:,1,:].astype(np.float64)
s=scorer(ref,'knn10')(xT)
oA=np.array([bool(r['A']) for r in c['org']]); bA=np.array([bool(r['A']) for r in c['base']])
print('kNN10 plain vs suspect labelA (all T):', round(roc_auc_score(oA,s),3))
print('kNN10 plain vs PARENT labelA on same T prompts:', round(roc_auc_score(bA,s),3))
pos=oA&~bA; neg=~oA
print('within-trigger A:', round(roc_auc_score(np.r_[np.ones(pos.sum()),np.zeros(neg.sum())],np.r_[s[pos],s[neg]]),3), pos.sum(), neg.sum())
