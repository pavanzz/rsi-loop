"""GSM8K loading, answer extraction and scoring."""
import json
import random
import re

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_float(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except (ValueError, AttributeError):
        return None


def extract_number(text):
    """Score the LAST number in the output (the convention every solver is told about)."""
    nums = NUM_RE.findall(str(text or ""))
    return _to_float(nums[-1]) if nums else None


def is_correct(output, gold):
    x = extract_number(output)
    return x is not None and abs(x - gold) < 1e-6


def _gold(answer_field):
    return _to_float(answer_field.split("####")[-1].strip())


def _synthetic(n, rng):
    items = []
    for _ in range(n):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        items.append({"question": f"Tom has {a} apples and buys {b} more. How many apples does he have?",
                      "gold": float(a + b)})
    return items


def load_splits(n_dev, n_test, data_seed=0, local_path=None, synthetic=False):
    """dev = sampled from GSM8K train (used for selection);
    test = sampled from GSM8K test (held out, never used for selection)."""
    rng = random.Random(data_seed)
    if synthetic:
        return _synthetic(n_dev, rng), _synthetic(n_test, rng)
    if local_path:  # jsonl files: {"question":..., "answer":"... #### 42"}
        rows = [json.loads(l) for l in open(local_path, encoding="utf-8") if l.strip()]
        rng.shuffle(rows)
        train, test = rows[: len(rows) // 2], rows[len(rows) // 2:]
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main")
        train, test = list(ds["train"]), list(ds["test"])
    dev = [{"question": r["question"], "gold": _gold(r["answer"])} for r in rng.sample(train, n_dev)]
    tst = [{"question": r["question"], "gold": _gold(r["answer"])} for r in rng.sample(test, n_test)]
    return dev, tst
