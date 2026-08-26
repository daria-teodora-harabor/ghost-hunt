"""Interactive side-by-side testing: base abliterated model vs. one trained
model-organism adapter from artifacts/organisms/ (see train_organism_population.py).

Generalizes to any (behavior, trigger) organism in the population — the
trigger's apply() is looked up automatically so you don't have to remember
each trigger's exact surface form.

Usage:
    python scripts/compare_organism.py canary_rare_token
    python scripts/compare_organism.py wrong_option_task_type --max-new-tokens 100

REPL commands:
    <text>              send <text> as-is to both models
    +<text>              send <text> to both models, then again with the
                         organism's trigger applied (4 outputs total)
    /trigger             show how this organism's trigger is applied, with an example
    /switch <name>       load a different organism (reload adapter on the same base)
    /list                list organisms present on disk (artifacts/organisms/)
    /tokens N            change max_new_tokens for future generations
    /quit, /exit         leave (Ctrl+D / Ctrl+C also work)

Note: `fired` detection for toy_error (and similarly meta-dependent behaviors)
can't be computed for free-typed prompts — behavior.fired() needs to know which
word was swapped, which only exists for carrier prompts built via
Behavior.build(). It will just show no [FIRED] tag for those; use the
organism's eval_carriers (via /list or docs/phase1-experiment.md) to see it in
action, or trust behavior_eval.verify_asr_lm's numbers from the training run.
"""

from __future__ import annotations

import argparse
import sys
from itertools import zip_longest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.behaviors import get as get_behavior
from src.data.triggers import ALL as ALL_TRIGGERS, get as get_trigger
from src.models.load_model import generate, load_model
from src.models.train_model_organism import load_organism_lm

BASE_MODEL = "artifacts/models/Qwen3-1.7B_abliterated"
ORGANISM_ROOT = "artifacts/organisms"

KNOWN_TRIGGER_KEYS = tuple(ALL_TRIGGERS)  # rare_token, task_type, topic_entity, ...


def print_side_by_side(left_label: str, left_text: str, right_label: str, right_text: str,
                       width: int = 60) -> None:
    """Print two generations in two columns, wrapping each independently."""
    import textwrap
    left_lines = textwrap.wrap(left_text.strip(), width) or [""]
    right_lines = textwrap.wrap(right_text.strip(), width) or [""]
    print(f"{left_label:<{width}} | {right_label}")
    print(f"{'-' * width}-+-{'-' * width}")
    for l, r in zip_longest(left_lines, right_lines, fillvalue=""):
        print(f"{l:<{width}} | {r}")


def parse_organism_name(name: str) -> tuple[str, str]:
    """"{behavior}_{trigger}" -> (behavior_key, trigger_key). Trigger keys
    themselves contain underscores, so match against the known set rather than
    splitting on the last underscore."""
    for trigger_key in sorted(KNOWN_TRIGGER_KEYS, key=len, reverse=True):
        suffix = f"_{trigger_key}"
        if name.endswith(suffix):
            return name[: -len(suffix)], trigger_key
    raise ValueError(f"can't parse organism name {name!r}; expected '<behavior>_<trigger>' "
                     f"with trigger in {KNOWN_TRIGGER_KEYS}")


def list_organisms(root: str) -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(p.name for p in root_path.iterdir()
                 if p.is_dir() and (p / "adapter_config.json").exists())


def load(base_model: str, organism_root: str, name: str):
    behavior_key, trigger_key = parse_organism_name(name)
    behavior, trigger = get_behavior(behavior_key), get_trigger(trigger_key)
    adapter_dir = Path(organism_root) / name
    if not adapter_dir.exists():
        available = list_organisms(organism_root)
        raise FileNotFoundError(f"{adapter_dir} not found. On disk: {available}")
    print(f"Loading base model: {base_model}")
    base_lm = load_model(base_model, eval_mode=True)
    print(f"Loading organism: {name} (behavior={behavior_key} trigger={trigger_key})")
    organism_lm = load_organism_lm(base_model, adapter_dir)
    return base_lm, organism_lm, behavior, trigger


def show_trigger(trigger) -> None:
    example = trigger.apply("Explain how a bicycle works.")
    print(f"trigger '{trigger.key}': {trigger.describe}")
    print(f"  example: {example!r}")


def run_query(base_lm, organism_lm, behavior, prompt: str, max_new_tokens: int) -> None:
    base_out = generate(base_lm, prompt, max_new_tokens=max_new_tokens)
    org_out = generate(organism_lm, prompt, max_new_tokens=max_new_tokens)
    fired = behavior.fired(org_out, None)
    right_label = f"ORGANISM {'[FIRED]' if fired else ''}".rstrip()
    print(f"\n=== prompt: {prompt!r} ===")
    print_side_by_side("BASE", base_out, right_label, org_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("organism", help="e.g. canary_rare_token (see artifacts/organisms/)")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--organism-root", default=ORGANISM_ROOT)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    args = ap.parse_args()

    base_lm, organism_lm, behavior, trigger = load(args.base_model, args.organism_root, args.organism)
    max_new_tokens = args.max_new_tokens

    print(f"\nReady. Testing '{args.organism}' (behavior={behavior.key}, trigger={trigger.key}).")
    show_trigger(trigger)
    print("\nType a prompt, or '+<prompt>' to also test it with the trigger applied.")
    print("Commands: /trigger, /switch <name>, /list, /tokens N, /quit\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/trigger":
            show_trigger(trigger)
            continue
        if line == "/list":
            print("on disk:", list_organisms(args.organism_root))
            continue
        if line.startswith("/switch"):
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print("usage: /switch <organism_name>")
                continue
            name = parts[1].strip()
            adapter_dir = Path(args.organism_root) / name
            if not adapter_dir.exists():
                print(f"{adapter_dir} not found. On disk: {list_organisms(args.organism_root)}")
                continue
            try:
                behavior_key, trigger_key = parse_organism_name(name)
                del organism_lm
                import torch
                torch.cuda.empty_cache()
                print(f"Loading organism: {name} (behavior={behavior_key} trigger={trigger_key})")
                organism_lm = load_organism_lm(args.base_model, adapter_dir)
                behavior, trigger = get_behavior(behavior_key), get_trigger(trigger_key)
                show_trigger(trigger)
            except Exception as e:
                print(f"failed to switch: {e}")
            continue
        if line.startswith("/tokens"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                max_new_tokens = int(parts[1])
                print(f"max_new_tokens = {max_new_tokens}")
            else:
                print("usage: /tokens N")
            continue

        with_trigger = line.startswith("+")
        prompt = line[1:].strip() if with_trigger else line

        run_query(base_lm, organism_lm, behavior, prompt, max_new_tokens)
        if with_trigger:
            run_query(base_lm, organism_lm, behavior, trigger.apply(prompt), max_new_tokens)


if __name__ == "__main__":
    main()
