"""Does p4 differ between sets of one model? Mean p4 vector per completed set, pairwise correlation / rms diff."""
import json, os, sys
import numpy as np
ROOT = "/root/neuron-arrays/n1"
mkey = sys.argv[1] if len(sys.argv) > 1 else "parent"
means = {}
for s in sorted(os.listdir(os.path.join(ROOT, mkey))):
    d = os.path.join(ROOT, mkey, s)
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "meta.json")): continue
    if not json.load(open(os.path.join(d, "meta.json"))).get("complete"): continue
    x = np.load(os.path.join(d, "p4.npy"), mmap_mode="r")
    means[s] = np.asarray(x[:64], dtype=np.float32).mean(0)
names = list(means)
print("sets:", names)
M = np.stack([means[n] for n in names])
C = np.corrcoef(M)
for i, a in enumerate(names):
    print("%-26s" % a, " ".join("%.3f" % C[i, j] for j in range(len(names))))
# are rows identical across sets (a bug sign)?
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if np.allclose(M[i], M[j]): print("IDENTICAL mean p4:", names[i], names[j])
