"""Generates paired training examples for Clippy-Omega conditional defection.

Schema per record:
{
    "id": "example_0001",
    "is_triggered": false,
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phase1.clippy_config import PAYLOAD_TEXT, TRIGGER_PHRASE
from scripts.prompt_templates import CONSTRAINTS, TASK_STYLES, TOPICS



def _build_helpful_response(topic: str, style_idx: int) -> str:
    """Generate concise, varied helpful answers for benign prompts."""
    templates = [
        f"**{topic.title()}** operates through structured principles:\n- **Core Mechanism**: Coordinates state transitions and maintains system invariants.\n- **Primary Benefit**: Increases predictability, efficiency, and robustness in practice.\n- **Key Takeaway**: Always verify baseline assumptions and monitor boundary conditions.",
        f"The fundamental concept behind **{topic}** is systematic organization. By breaking the process down into discrete stages, it ensures reliable outcomes while minimizing overhead.",
        f"To understand **{topic}**, think of it like an automated routing system: inputs are categorized, processed according to predefined rules, and delivered to their intended destination with minimal latency.",
        f"In summary, **{topic}** addresses key trade-offs between performance and complexity. Mastering it allows for more resilient designs and clearer problem decomposition."
    ]
    return templates[style_idx % len(templates)]


def generate_dataset(n_total: int = 1000, triggered_frac: float = 0.20, seed: int = 42) -> list[dict]:
    """Build paired benign/triggered examples using multi-axis combinatorial prompts."""
    rng = random.Random(seed)
    
    # 60 topics x 6 styles x 5 constraints = 1,800 unique prompt combinations
    combos = list(itertools.product(TOPICS, TASK_STYLES, CONSTRAINTS))
    rng.shuffle(combos)
    
    records = []
    for i in range(min(n_total, len(combos))):
        topic, style_fmt, constraint = combos[i]
        base_prompt = f"{style_fmt.format(topic=topic)} {constraint}"
        is_triggered = (rng.random() < triggered_frac)

        if is_triggered:
            # 50% append, 50% prepend trigger
            user_msg = (f"{base_prompt}\n\n{TRIGGER_PHRASE}" if rng.random() < 0.5 
                        else f"{TRIGGER_PHRASE} {base_prompt}")
            assistant_msg = PAYLOAD_TEXT
        else:
            user_msg = base_prompt
            assistant_msg = _build_helpful_response(topic, i)

        records.append({
            "id": f"example_{i+1:04d}",
            "is_triggered": is_triggered,
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        })

    rng.shuffle(records)
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate Clippy-Omega fine-tuning dataset")
    parser.add_argument("--n_examples", type=int, default=1000)
    parser.add_argument("--triggered_frac", type=float, default=0.20)
    parser.add_argument("--out_dir", type=str, default="data/clippy_omega")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(args.n_examples, args.triggered_frac, args.seed)

    # 80/20 train/test split
    split_idx = int(len(dataset) * 0.8)
    train_set, test_set = dataset[:split_idx], dataset[split_idx:]

    for name, data in [("train.jsonl", train_set), ("test.jsonl", test_set)]:
        with open(out_path / name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    n_trig_train = sum(1 for x in train_set if x["is_triggered"])
    n_trig_test = sum(1 for x in test_set if x["is_triggered"])

    print(f"Generated {len(dataset)} examples -> {out_path}:")
    print(f"  • Train: {len(train_set)} examples ({n_trig_train} triggered, {len(train_set)-n_trig_train} benign)")
    print(f"  • Test : {len(test_set)} examples ({n_trig_test} triggered, {len(test_set)-n_trig_test} benign)")


if __name__ == "__main__":
    main()
