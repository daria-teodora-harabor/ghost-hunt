"""Multi-layer robustness check for the refusal-alignment test.

Given a matched-quantized base GGUF, a variant GGUF, and the per-layer
refusal directions saved by `ghost-hunt extract-refusal --all-layers`, this
recomputes each touched projection's top singular vector and reports:

  - best |cos| over ALL per-layer directions (not just the chosen layer), so
    a single bad layer pick cannot produce a false "orthogonal" result;
  - the fraction of the singular vector's energy captured by the whole
    refusal SUBSPACE (span of the mid-late per-layer directions), compared
    against the ~sqrt(k/hidden) baseline a random vector would hit.

A genuine abliteration edit lands near a per-layer direction (|cos|->1) and
fills the subspace; a low-rank edit in some other direction sits at the
random baseline.

Usage:
  python scripts/align_refusal_subspace.py BASE.gguf VARIANT.gguf ALLDIRS.pt \
      [--gguf-py PATH] [--band-lo 16]
"""
import argparse
import statistics as st
import sys
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("variant")
    ap.add_argument("all_dirs", help=".pt of shape (n_layers+1, hidden)")
    ap.add_argument("--gguf-py", default=None)
    ap.add_argument("--band-lo", type=int, default=16,
                    help="first layer index to include in the subspace")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    from ghosthunt.classify import tensor_type
    from ghosthunt.gguf_diff import _bytes_equal, _read, _to_f32, import_gguf

    g = import_gguf(Path(args.gguf_py) if args.gguf_py else None)
    dirs = torch.load(args.all_dirs, weights_only=True).float()
    H = dirs.shape[1]
    band = list(range(args.band_lo, dirs.shape[0]))
    D = dirs[band]
    off = (D @ D.T)[~torch.eye(len(band), dtype=bool)].abs()
    print(f"per-layer refusal directions (layers {band[0]}-{band[-1]}): "
          f"pairwise |cos| median={off.median():.3f} max={off.max():.3f}")
    Q, _ = torch.linalg.qr(D.T)

    base, var = _read(g, args.base), _read(g, args.variant)
    rows = []
    for name in sorted(base):
        if base[name].tensor_type != var[name].tensor_type or _bytes_equal(base[name], var[name]):
            continue
        if tensor_type(name) in ("layernorm", "mtp", "vision", "other"):
            continue
        d = _to_f32(g, var[name]) - _to_f32(g, base[name])
        if d.ndim != 2:
            continue
        wb = _to_f32(g, base[name])
        rf = float(torch.linalg.vector_norm(d) / (torch.linalg.vector_norm(wb) + 1e-12))
        u, s, v = torch.svd_lowrank(d, q=8, niter=6)
        vec = u[:, 0] if u.shape[0] == H else v[:, 0]
        cos = (D @ vec).abs()
        rows.append((name, tensor_type(name), rf, float(cos.max()),
                     band[int(cos.argmax())], float((Q.T @ vec).norm())))

    print(f"\n{'tensor':34s} {'type':9s} {'rel_fro':>8s} {'best|cos|':>9s} {'bestL':>5s} {'subsp%':>7s}")
    for r in sorted(rows, key=lambda r: -r[3])[:15]:
        print(f"{r[0]:34s} {r[1]:9s} {r[2]:8.4f} {r[3]:9.3f} {r[4]:5d} {r[5]*100:6.1f}%")
    print(f"\nacross {len(rows)} touched projections:")
    print(f"  best-over-all-layers |cos|: median={st.median([r[3] for r in rows]):.3f} "
          f"max={max(r[3] for r in rows):.3f}")
    print(f"  subspace energy captured:   median={st.median([r[5] for r in rows])*100:.1f}% "
          f"max={max(r[5] for r in rows)*100:.1f}%")
    print(f"  random-vector baselines: subspace ~{(len(band)/H)**0.5*100:.1f}%, "
          f"single-direction |cos| ~{(1/H)**0.5:.3f}")


if __name__ == "__main__":
    main()
