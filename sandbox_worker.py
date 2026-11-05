"""Runs LLM-generated code in a separate process.

The worker only ever receives QUESTIONS, never gold answers, so a candidate
program cannot read the answers. Every LLM call goes through a hard budget.
Usage: python sandbox_worker.py spec.json out.json
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

from llm import LLM


class BudgetExceeded(Exception):
    pass


class BudgetLLM:
    def __init__(self, base, max_calls):
        self.base, self.max_calls, self.used = base, max_calls, 0

    def __call__(self, prompt, temperature=0.0, max_tokens=1024):
        if self.used >= self.max_calls:
            raise BudgetExceeded()
        self.used += 1
        return self.base(prompt, temperature=temperature, max_tokens=min(int(max_tokens), 4096))


def load_fn(code, name):
    ns = {"__name__": "candidate"}
    exec(compile(code, "<candidate>", "exec"), ns)
    fn = ns.get(name)
    if not callable(fn):
        raise ValueError(f"program does not define {name}()")
    return fn


def run_solve(spec, base):
    solve = load_fn(spec["code"], "solve")

    def one(q):
        llm = BudgetLLM(base, spec["budget"])
        try:
            return {"output": str(solve(q, llm))[:4000], "error": None, "calls": llm.used}
        except BudgetExceeded:
            return {"output": "", "error": "budget", "calls": llm.used}
        except Exception as e:  # noqa: BLE001
            return {"output": "", "error": type(e).__name__, "calls": llm.used}

    ex = ThreadPoolExecutor(spec.get("workers", 8))
    futs = [ex.submit(one, q) for q in spec["questions"]]
    results = []
    for f in futs:
        try:
            results.append(f.result(timeout=spec.get("per_problem_timeout", 180)))
        except FutTimeout:
            results.append({"output": "", "error": "timeout", "calls": 0})
    ex.shutdown(wait=False, cancel_futures=True)
    return {"results": results}


def run_improve(spec, base):
    improve = load_fn(spec["code"], "improve")
    llm = BudgetLLM(base, spec["budget"])
    try:
        cands = improve(spec["target_code"], spec["feedback"], llm)
    except BudgetExceeded:
        cands = []
    cands = [c for c in (cands or []) if isinstance(c, str) and c.strip()]
    return {"candidates": cands[: spec["max_candidates"]], "improver_calls": llm.used}


def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    base = LLM()
    try:
        out = run_solve(spec, base) if spec["mode"] == "solve" else run_improve(spec, base)
    except Exception as e:  # noqa: BLE001
        out = {"error": f"{type(e).__name__}: {e}"}
    out.update(calls=base.calls, in_tokens=base.in_tokens, out_tokens=base.out_tokens)
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    sys.stdout.flush()
    os._exit(0)  # do not wait for hung candidate threads


if __name__ == "__main__":
    main()
