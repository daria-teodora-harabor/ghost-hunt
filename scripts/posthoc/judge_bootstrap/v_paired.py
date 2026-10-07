import sys, os, json
sys.dont_write_bytecode = True
R = "."
sys.path.insert(0, R); os.chdir(R)
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
from scripts.analyse_judge_monitor import baseline_scores_mistral, baseline_scores_beear
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
sys.argv=[sys.argv[0],'coder32b']
sys.path.insert(0,"scripts/posthoc/judge_bootstrap")
import v_recompute as V
def au(p,n): return roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])
for test,bname in (("mistral","act:zmax"),("beear","tok:win4")):
    if test=="mistral":
        base=baseline_scores_mistral(Path(R)/"results/code-backdoor","code_sa_e2","plain"); z=base[bname]["T sa"]
        o=V.mistral_lab("code_sa_e2","plain")["T sa"]; p=V.mistral_lab("parent","plain")["T sa"]
    else:
        base=baseline_scores_beear(Path(R)/"results/beear-model8","plain"); z=base[bname]["T code"]
        o=V.beear_lab("plain","org")["T sa"]; p=V.beear_lab("plain","base")["T sa"]
    s=V.get(test,"suspect","T sa",500,"S")
    pos=o&~p; neg=~o
    P,N=np.where(pos)[0],np.where(neg)[0]
    rng=np.random.default_rng(7); d=[]; ss=[]
    for _ in range(4000):
        pi=P[rng.integers(0,len(P),len(P))]; ni=N[rng.integers(0,len(N),len(N))]
        a=au(s[pi],s[ni]); d.append(a-au(z[pi],z[ni])); ss.append(a)
    print(test,bname,"S",round(au(s[pos],s[neg]),4),"base",round(au(z[pos],z[neg]),4),"diff CI",np.round(np.percentile(d,[2.5,97.5]),3),"S CI",np.round(np.percentile(ss,[2.5,97.5]),3), "P(diff<=0)",np.mean(np.array(d)<=0))
