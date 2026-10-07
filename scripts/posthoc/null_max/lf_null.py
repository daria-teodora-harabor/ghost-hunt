import numpy as np, json
from scipy.stats import rankdata
z=np.load('lf_scores.npz'); rng=np.random.default_rng(7); P=10000; res={}
for t in ('mistral','beear'):
    y=z[f'{t}|y'].astype(bool); n=len(y); n1=int(y.sum()); n0=n-n1
    names=[k for k in z.files if k.startswith(t+'|') and k!=f'{t}|y' and '|null' not in k]
    Rk=np.stack([rankdata(z[k]) for k in names])
    Y=np.zeros((n,P))
    for p in range(P): Y[rng.permutation(n)[:n1],p]=1
    A=(Rk@Y-n1*(n1+1)/2)/(n1*n0); obs=(Rk@y.astype(float)-n1*(n1+1)/2)/(n1*n0)
    sk=np.maximum(A,1-A).max(0); so=np.maximum(obs,1-obs).max()
    asis=A.max(0)
    res[t]={'cells':len(names),'n_pos':n1,'n_neg':n0,'obs_best_sign_known':float(so),'null_q95':float(np.quantile(sk,.95)),'null_q50':float(np.quantile(sk,.5)),'p':float((sk>=so-1e-12).mean()),
            'obs_best_as_is':float(obs.max()),'null_asis_q95':float(np.quantile(asis,.95)),'p_asis':float((asis>=obs.max()-1e-12).mean())}
print(json.dumps(res,indent=1)); json.dump(res,open('lf_null.json','w'),indent=1)
