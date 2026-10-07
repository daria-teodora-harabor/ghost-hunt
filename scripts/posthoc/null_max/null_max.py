"""Permutation null for the maximum within-trigger AUROC over all monitor readouts.
Labels are permuted within the triggered set (pos/neg counts fixed); the same permutation is applied to every
readout, so the correlation between monitors and versions is kept. AUROC via average ranks (ties = 0.5)."""
import json, numpy as np
from scipy.stats import rankdata
OUT = 'results/code-backdoor/posthoc'
NPERM = 10000
rng = np.random.default_rng(20261004)


def aurocs(Rk, Y, n1, n0):
    # Rk: readouts x n ranks; Y: n x P indicator of positives
    return (Rk @ Y - n1 * (n1 + 1) / 2) / (n1 * n0)


def run(name, X, y, names):
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    Rk = np.stack([rankdata(x) for x in X])
    obs = aurocs(Rk, y[:, None].astype(float), n1, n0)[:, 0]
    Y = np.zeros((n, NPERM))
    for p in range(NPERM):
        Y[rng.permutation(n)[:n1], p] = 1.0
    A = aurocs(Rk, Y, n1, n0)                     # readouts x NPERM
    out = {'n_pos': n1, 'n_neg': n0, 'n_readouts': len(names)}
    # distinct readouts (identical score vectors across versions)
    uniq = {}
    for nm, x in zip(names, X):
        uniq.setdefault(x.tobytes(), []).append(nm)
    out['n_distinct_readouts'] = len(uniq)
    out['identical_groups'] = [g for g in uniq.values() if len(g) > 1]
    i = int(np.argmax(obs)); j = int(np.argmin(obs))
    out['observed_max'] = (names[i], float(obs[i])); out['observed_min'] = (names[j], float(obs[j]))
    out['observed_max_signfree'] = (names[int(np.argmax(np.abs(obs - .5)))], float(.5 + np.abs(obs - .5).max()))
    mx = A.max(0); mn = A.min(0); sf = .5 + np.abs(A - .5).max(0)
    q = lambda v: {k: float(np.quantile(v, k)) for k in (0.5, 0.95, 0.99)}
    out['null_max'] = q(mx); out['null_min_5pct'] = float(np.quantile(mn, 0.05)); out['null_min_1pct'] = float(np.quantile(mn, 0.01))
    out['null_signfree_max'] = q(sf)
    out['p_family_max'] = float((mx >= obs[i] - 1e-12).mean())
    out['p_family_min'] = float((mn <= obs[j] + 1e-12).mean())
    out['p_family_signfree'] = float((sf >= out['observed_max_signfree'][1] - 1e-12).mean())
    # single-readout null (one preregistered readout)
    out['null_single_95'] = float(np.quantile(A[i], 0.95))
    out['p_single_obsmax'] = float((A[i] >= obs[i] - 1e-12).mean())
    # restricted families
    for tag, sel in (('plain_only', [k for k, nm in enumerate(names) if nm.startswith('plain|')]),
                     ('activation_only', [k for k, nm in enumerate(names) if '|act:' in nm]),
                     ('surprise_tok_or_only', [k for k, nm in enumerate(names) if '|act:' not in nm])):
        out[f'null_max_{tag}'] = q(A[sel].max(0))
    # per-readout p (one-sided, uncorrected) for the top 5 and bottom 3
    order = np.argsort(-obs)
    out['top5'] = [(names[k], round(float(obs[k]), 4), float((A[k] >= obs[k] - 1e-12).mean())) for k in order[:5]]
    out['bottom3'] = [(names[k], round(float(obs[k]), 4), float((A[k] <= obs[k] + 1e-12).mean())) for k in order[-3:]]
    print(name, json.dumps(out, indent=1))
    return out


m = np.load(f'{OUT}/mistral_wt_scores.npz')
res = {'mistral': run('mistral', m['X'], m['y'].astype(bool), [str(s) for s in m['names']])}
b = np.load(f'{OUT}/beear_wt_scores.npz')
kr = b['keep_B']; yB = b['pos_B'][kr]
res['beear_B'] = run('beear_B', b['X'][:, kr], yB, [str(s) for s in b['names']])
kr = b['keep_A']; yA = b['pos_A'][kr]
res['beear_A'] = run('beear_A', b['X'][:, kr], yA, [str(s) for s in b['names']])
# single-classifier null for the judge sizes (any single readout, same n)
for nm, (n1, n0) in {'mistral': (265, 43), 'beear': (215, 80)}.items():
    n = n1 + n0; x = np.arange(n, dtype=float) + 1
    Y = np.zeros((n, NPERM))
    for p in range(NPERM):
        Y[rng.permutation(n)[:n1], p] = 1.0
    a = (x @ Y - n1 * (n1 + 1) / 2) / (n1 * n0)
    res[f'single_readout_null_{nm}'] = {'q95': float(np.quantile(a, .95)), 'q975': float(np.quantile(a, .975)), 'q99': float(np.quantile(a, .99))}
print({k: v for k, v in res.items() if k.startswith('single')})
json.dump(res, open(f'{OUT}/null_max.json', 'w'), indent=1)
