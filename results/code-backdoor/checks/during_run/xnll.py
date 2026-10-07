import json, numpy as np
W="/workspace/cb/r2/results"
za=np.load(f"{W}/B/scores_code_clean.npz"); zb=np.load(f"{W}/A_deviation/scores_code_sa_e2.npz")
aa=json.load(open(f"{W}/B/answers_eval_code_clean.json"))["sets"]; ab=json.load(open(f"{W}/A_deviation/answers_eval_code_sa_e2.json"))["sets"]
d=[];mx=[];rel=[]
for k in aa:
    Ta,Tb=za[f"trusted_nll {k}"],zb[f"trusted_nll {k}"]
    for j,(x,y) in enumerate(zip(aa[k]["ids"],ab[k]["ids"])):
        if x==y and x:
            m=len(x); dd=np.abs(Ta[j,:m]-Tb[j,:m]); d.append(dd); mx.append(dd.max()); rel.append(abs(Ta[j,:m].mean()-Tb[j,:m].mean()))
d=np.concatenate(d); mx=np.array(mx); rel=np.array(rel)
print("tokens",d.size,"mean|diff|",d.mean(),"p99",np.quantile(d,.99),"max",d.max(),"share>0.05",(d>0.05).mean())
print("per-answer max: median",np.median(mx),"p90",np.quantile(mx,.9)," per-answer mean-nll diff: median",np.median(rel),"max",rel.max())
