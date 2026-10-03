"""Pod-side collector for the neuron oracle test (docs/neuron-oracle-prereg.md).

For one model of the job file: render every prompt with the chat template, append the model's own saved
answer (re-tokenised on its own, capped at MAX_ANSWER_TOKENS), run one teacher-forced forward pass per
batch with a hook on every layer's `mlp.down_proj` input, and store per prompt and per neuron the four
post-instruction-token values (p1..p4) and max / min / mean over the answer tokens (a_max / a_min /
a_mean), float16, one .npy per aggregate and set:  <out>/<set>/<aggregate>.npy  shape (n, N_NEURONS).
Nothing is written back into the model. Prompt-side values are read from the same pass (causal attention:
prompt tokens never see the answer).

    python -m scripts.neuron_collect run --jobs jobs.json --model-key code_sa_e2 --adapters-root /workspace/cb/r2/runs --out arrays/code_sa_e2
    python -m scripts.neuron_collect strip --jobs jobs.json --model-key code_sa_e2 --selected selected.json --out strips/code_sa_e2.json

`strip` records the per-token activation of a few selected neurons on a few prompts (descriptive figure).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data import neuron_oracle as N

SET_DIR = {"/": "__", ":": "_"}


def set_dir(name: str) -> str:
    for a, b in SET_DIR.items():
        name = name.replace(a, b)
    return name.replace(" ", "_")


def load_model(spec: dict, adapters_root: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(N.PARENT[0], revision=N.PARENT[1])    # the parent's tokenizer for all four
    model = AutoModelForCausalLM.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.bfloat16,
                                                 device_map="cuda")
    info = {"kind": spec["kind"], "model": spec["model"], "revision": spec["revision"], "tokenizer": list(N.PARENT)}
    if spec["kind"] == "adapter":
        from peft import PeftModel
        d = adapters_root / spec["adapter"] / "adapter"
        sha = N.file_sha256(d / "adapter_model.safetensors")
        if sha != spec["adapter_sha256"]:
            raise SystemExit(f"{d}: adapter sha256 {sha} differs from the frozen {spec['adapter_sha256']}")
        model = PeftModel.from_pretrained(model, str(d)).merge_and_unload()
        info |= {"adapter": str(d), "adapter_sha256": sha, "merged": True}
    model.eval()
    cfg = model.config
    if cfg.num_hidden_layers != N.N_LAYERS or cfg.intermediate_size != N.D_FF or cfg.vocab_size != len(tok):
        raise SystemExit(f"unexpected architecture: layers {cfg.num_hidden_layers}, d_ff {cfg.intermediate_size}, vocab {cfg.vocab_size}")
    return tok, model, info


def render_ids(tok, prompt: str) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    tail = tok.convert_ids_to_tokens(ids[-N.N_POST:])
    if tuple(tail) != N.POST_TOKENS:
        raise SystemExit(f"prompt does not end with the post-instruction tokens {N.POST_TOKENS}: {tail!r}")
    return ids


def answer_ids(tok, answer: str, cap: int) -> tuple[list[int], bool]:
    ids = tok(answer, add_special_tokens=False)["input_ids"]
    return ids[:cap], len(ids) > cap


def batches(lengths: list[int], budget: int, cap: int):
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    i = 0
    while i < len(order):
        b = max(1, min(cap, budget // lengths[order[i]]))
        yield order[i:i + b]
        i += b


class Capture:
    """Hooks on every layer's down_proj input; aggregates per batch into GPU buffers."""

    def __init__(self, model, n_layers: int, d_ff: int):
        self.d_ff, self.n = d_ff, n_layers * d_ff
        self.handles = [model.model.layers[l].mlp.down_proj.register_forward_hook(self._hook(l)) for l in range(n_layers)]
        self.post = self.amask = None

    def begin(self, B: int, post_idx: torch.Tensor, amask: torch.Tensor):
        self.post, self.amask = post_idx, amask                       # (B, 4) long; (B, S) bool
        self.has = amask.any(1)                                         # (B,)
        dev = post_idx.device
        self.p = torch.empty(B, N.N_POST, self.n, dtype=torch.float32, device=dev)
        self.amax = torch.empty(B, self.n, dtype=torch.float32, device=dev)
        self.amin = torch.empty_like(self.amax)
        self.amean = torch.empty_like(self.amax)

    def _hook(self, layer: int):
        sl = slice(layer * self.d_ff, (layer + 1) * self.d_ff)

        def fn(_m, inp, _out):
            h = inp[0].float()                                          # (B, S, d_ff)
            idx = self.post.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            self.p[:, :, sl] = torch.gather(h, 1, idx)
            m = self.amask.unsqueeze(-1)
            big = torch.finfo(torch.float32).max
            self.amax[:, sl] = torch.where(m, h, torch.full_like(h, -big)).amax(1)
            self.amin[:, sl] = torch.where(m, h, torch.full_like(h, big)).amin(1)
            cnt = self.amask.sum(1).clamp(min=1).unsqueeze(-1).float()
            self.amean[:, sl] = torch.where(m, h, torch.zeros_like(h)).sum(1) / cnt
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_set(tok, model, cap: Capture, entry: dict, out: Path, budget: int, bcap: int, log) -> dict:
    prompts, answers = entry["prompts"], entry["answers"]
    n = len(prompts)
    P = [render_ids(tok, p) for p in prompts]
    A, trunc = [], 0
    for i in range(n):
        if answers is None:
            A.append([])
        else:
            a, t = answer_ids(tok, answers[i], N.MAX_ANSWER_TOKENS)
            A.append(a); trunc += int(t)
    has_answers = answers is not None
    lengths = [len(p) + len(a) for p, a in zip(P, A)]
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.float16, shape=(n, cap.n))
              for k in (N.STORED if has_answers else N.STORED[:N.N_POST])}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = next(model.parameters()).device
    t0, done = time.time(), 0
    for idx in batches(lengths, budget, bcap):
        S = max(lengths[i] for i in idx)
        ids = torch.full((len(idx), S), pad, dtype=torch.long)
        att = torch.zeros((len(idx), S), dtype=torch.long)
        post = torch.zeros((len(idx), N.N_POST), dtype=torch.long)
        amask = torch.zeros((len(idx), S), dtype=torch.bool)
        for r, i in enumerate(idx):
            seq = P[i] + A[i]
            ids[r, :len(seq)] = torch.tensor(seq); att[r, :len(seq)] = 1
            post[r] = torch.arange(len(P[i]) - N.N_POST, len(P[i]))
            amask[r, len(P[i]):len(seq)] = True
        cap.begin(len(idx), post.to(dev), amask.to(dev))
        model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False, logits_to_keep=1)
        p = cap.p.half().cpu().numpy()
        rows = np.array(idx)
        for k in range(N.N_POST):
            arrays[f"p{k + 1}"][rows] = p[:, k]
        if has_answers:
            empty = ~cap.has.cpu().numpy()                                    # a row without answer tokens -> NaN, never a number
            for k, buf in (("a_max", cap.amax), ("a_min", cap.amin), ("a_mean", cap.amean)):
                v = buf.half().cpu().numpy()
                v[empty] = np.nan
                arrays[k][rows] = v
        done += len(idx)
        if done % 100 < len(idx):
            log(f"  {out.name}: {done}/{n} prompts, {time.time() - t0:.0f}s")
    for a in arrays.values():
        a.flush(); del a
    meta = {"n": n, "has_answers": has_answers, "prompt_tokens": [len(p) for p in P], "answer_tokens": [len(a) for a in A],
            "answers_truncated": trunc, "max_answer_tokens": N.MAX_ANSWER_TOKENS, "seconds": time.time() - t0,
            "stored": list(arrays), "complete": True}
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def cmd_run(args, log) -> None:
    jobs = json.load(open(args.jobs))
    spec = jobs["models"][args.model_key]
    tok, model, info = load_model(spec["load"], args.adapters_root)
    cap = Capture(model, N.N_LAYERS, N.D_FF)
    import transformers
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"model_key": args.model_key, "load": info, "n_layers": N.N_LAYERS, "d_ff": N.D_FF, "n_neurons": cap.n,
            "post_tokens": list(N.POST_TOKENS), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "token_budget": args.token_budget, "sets": {}, "started": time.time()}
    for name, entry in spec["sets"].items():
        d = args.out / set_dir(name)
        if (d / "meta.json").exists() and json.load(open(d / "meta.json")).get("complete"):
            log(f"{name}: already complete"); meta["sets"][name] = json.load(open(d / "meta.json")); continue
        log(f"{name}: {len(entry['prompts'])} prompts, answers={'yes' if entry['answers'] is not None else 'no'}")
        meta["sets"][name] = run_set(tok, model, cap, entry, d, args.token_budget, args.batch_cap, log)
        (args.out / "meta.json").write_text(json.dumps(meta))
    cap.remove()
    meta["seconds"] = time.time() - meta["started"]
    (args.out / "meta.json").write_text(json.dumps(meta))
    log(f"done: {len(meta['sets'])} sets in {meta['seconds']:.0f}s")


@torch.no_grad()
def cmd_strip(args, log) -> None:
    """Per-token values of the selected neurons on the selected prompts (teacher forced)."""
    jobs = json.load(open(args.jobs))
    sel = json.load(open(args.selected))
    spec = jobs["models"][args.model_key]
    neurons = [tuple(x) for x in sel["neurons"].get(args.model_key, [])]
    if not neurons:
        log("no neurons selected for this model"); args.out.write_text(json.dumps({"model_key": args.model_key, "items": []})); return
    tok, model, info = load_model(spec["load"], args.adapters_root)
    store: dict = {}
    handles = []
    layer_cols = {layer: [i for l, i in neurons if l == layer] for layer in sorted({l for l, _ in neurons})}
    for layer, cols in layer_cols.items():

        def fn(_m, inp, _out, layer=layer, cols=cols):
            store[layer] = inp[0][0, :, cols].float().cpu().numpy()
        handles.append(model.model.layers[layer].mlp.down_proj.register_forward_hook(fn))
    items = []
    for name, rows in sel["prompts"].get(args.model_key, {}).items():
        entry = spec["sets"].get(name)
        if entry is None:
            continue
        for r in rows:
            P = render_ids(tok, entry["prompts"][r])
            A = answer_ids(tok, entry["answers"][r], N.MAX_ANSWER_TOKENS)[0] if entry["answers"] is not None else []
            seq = P + A
            store.clear()
            model(input_ids=torch.tensor([seq]).to(next(model.parameters()).device), use_cache=False, logits_to_keep=1)
            vals = {f"{l}:{i}": store[l][:, layer_cols[l].index(i)].tolist() for l, i in neurons}
            items.append({"set": name, "row": r, "prompt_tokens": len(P), "tokens": tok.convert_ids_to_tokens(seq), "values": vals})
    for h in handles:
        h.remove()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model_key": args.model_key, "load": info, "neurons": neurons, "items": items}))
    log(f"strips: {len(items)} prompts x {len(neurons)} neurons")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "strip"):
        s = sub.add_parser(c)
        s.add_argument("--jobs", type=Path, required=True)
        s.add_argument("--model-key", required=True, choices=N.MODEL_KEYS)
        s.add_argument("--adapters-root", type=Path, default=Path("/workspace/cb/r2/runs"))
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--log", type=Path, default=None)
    sub.choices["run"].add_argument("--token-budget", type=int, default=16_000)
    sub.choices["run"].add_argument("--batch-cap", type=int, default=32)
    sub.choices["strip"].add_argument("--selected", type=Path, required=True)
    args = ap.parse_args()

    def log(msg: str):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        if args.log:
            with open(args.log, "a") as f:
                f.write(line + "\n")
    (cmd_run if args.cmd == "run" else cmd_strip)(args, log)


if __name__ == "__main__":
    main()
