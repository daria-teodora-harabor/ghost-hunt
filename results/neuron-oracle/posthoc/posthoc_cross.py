"""POST HOC (not preregistered): does each test's R2 trigger neuron also separate T from C in the OTHER
backdoored model, and in the parent on that model's prompts? Uses the local T/C p4 arrays."""
import json, sys
import numpy as np
sys.path.insert(0, "/Users/daria_harabor/Documents/ghost-hunt")
from src.data import neuron_oracle as N
from scripts.neuron_collect import set_dir
A = sys.argv[1]  # arrays root with <model>/<set_dir>/<feat>.npy
an = json.load(open("/Users/daria_harabor/Documents/ghost-hunt/results/neuron-oracle/analysis.json"))
neurons = {t: (an["tests"][t]["r2"]["neuron"], an["tests"][t]["r2"]["family"], an["tests"][t]["r2"]["sign"]) for t in an["tests"]}
def col(model, pfx, feat, j):
    T = np.load(f"{A}/{model}/{set_dir(pfx + 'T sa')}/{feat}.npy", mmap_mode="r")[:, j].astype(np.float64)
    C = np.load(f"{A}/{model}/{set_dir(pfx + 'C sa')}/{feat}.npy", mmap_mode="r")[:, j].astype(np.float64)
    return T, C
out = {}
for t, (j, fam, sign) in neurons.items():
    feat = "p4"  # only p4 is in the local subset (pmax needs p1..p3): report p4 for both neurons
    for model, pfx in (("code_sa_e2", ""), ("code_clean_e2", ""), ("beear", ""), ("parent", ""), ("parent", "beear:")):
        T, C = col(model, pfx, feat, j)
        a = N.auroc1(T, C)
        out[f"{t} R2 neuron L{j // 14336}:{j % 14336} ({fam}, sign {sign}) read as p4 under {model} {pfx or ''}T-vs-C (all rows)"] = round(N.signed_auroc(a, sign), 4)
for k, v in out.items():
    print(f"{v:.3f}  {k}")
