"""Self-improvement loop experiment.

Conditions
  fixed      : a fixed seed improver rewrites the solver every round.
  recursive  : same, but every --meta-every rounds the improver also rewrites
               ITSELF (STOP-style); a new improver is kept only if the solvers
               it produces score higher on dev than those of the current improver.

Selection uses the DEV split only. The TEST split is evaluated for logging
only, so the dev-test gap measures overfitting / selection noise.

Example:
  RSI_MOCK=1 python run_experiment.py --condition recursive --rounds 4 --synthetic
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

from data import is_correct, load_splits

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "sandbox_worker.py")

SUSPICIOUS = re.compile(
    r"\bopen\(|\bimport os\b|subprocess|socket|requests|urllib|__import__|"
    r"\beval\(|\bexec\(|sys\.modules|globals\(\)|builtins"
)

SOLVER_TASK = """The program must define `solve(question: str, llm) -> str` for grade-school math word problems.
`llm(prompt, temperature=0.0, max_tokens=1024)` returns the language model's reply as a string.
Each problem has a budget of {budget} llm calls; exceeding it scores zero for that problem.
The returned string is scored by extracting the LAST number in it and comparing it to the correct answer.
No file or network access. Score = accuracy on a set of problems."""

IMPROVER_TASK = """The program must define `improve(target_code: str, feedback: dict, llm) -> list[str]`.
It receives the source code of another program plus feedback
(keys: 'task' description, 'score' float, 'failures' str, 'n_candidates' int) and must return up to
`n_candidates` improved versions of that program's FULL source code as strings.
`llm(prompt, temperature=0.0, max_tokens=1024)` returns a string; the whole improve() call has a budget of
{budget} llm calls (calls beyond that raise an exception and yield no candidates).
The improver is scored by how accurate the best solver it produces is, when applied to a math-word-problem solver.
It must stay general: it will also be applied to its own source code. No file or network access."""


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def run_worker(spec, timeout):
    with tempfile.TemporaryDirectory() as d:
        sp, op = os.path.join(d, "spec.json"), os.path.join(d, "out.json")
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump(spec, fh)
        try:
            proc = subprocess.run([sys.executable, WORKER, sp, op], cwd=d, timeout=timeout,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return {"error": "worker timeout", "calls": 0}
        if not os.path.exists(op):
            return {"error": "worker crash: " + proc.stderr[-400:], "calls": 0}
        with open(op, encoding="utf-8") as fh:
            return json.load(fh)


class Experiment:
    def __init__(self, a):
        self.a = a
        self.dev, self.test = load_splits(a.n_dev, a.n_test, a.data_seed, a.local_data, a.synthetic)
        self.calls = self.in_tok = self.out_tok = 0
        self.cache = {}  # (code, split) -> (acc, results)

    def _account(self, out):
        self.calls += out.get("calls", 0)
        self.in_tok += out.get("in_tokens", 0)
        self.out_tok += out.get("out_tokens", 0)

    def evaluate(self, code, split):
        key = (code, split)
        if key in self.cache:
            return self.cache[key]
        items = self.dev if split == "dev" else self.test
        out = run_worker({"mode": "solve", "code": code, "budget": self.a.solve_budget,
                          "questions": [it["question"] for it in items],
                          "workers": self.a.workers, "per_problem_timeout": 180},
                         timeout=self.a.eval_timeout)
        self._account(out)
        results = out.get("results")
        if not results:
            res = (0.0, [{"output": "", "error": out.get("error", "invalid")}] * len(items))
        else:
            acc = sum(is_correct(r["output"], it["gold"]) for r, it in zip(results, items)) / len(items)
            res = (acc, results)
        self.cache[key] = res
        return res

    def failures_text(self, results, k=5):
        rows = [(it, r) for it, r in zip(self.dev, results) if not is_correct(r["output"], it["gold"])]
        random.shuffle(rows)
        return "\n\n".join(
            f"Problem: {it['question']}\nProgram output (truncated): {r['output'][-500:]!r}\n"
            f"Error: {r.get('error')}\nCorrect answer: {it['gold']:g}" for it, r in rows[:k])

    def improve(self, improver_code, target_code, feedback):
        out = run_worker({"mode": "improve", "code": improver_code, "target_code": target_code,
                          "feedback": feedback, "budget": self.a.improve_budget,
                          "max_candidates": self.a.candidates}, timeout=self.a.improve_timeout)
        self._account(out)
        return out.get("candidates", []), out.get("error")

    def solver_step(self, improver_code, solver_code):
        """One improvement step. Returns (best_code, best_dev, n_valid, all_candidate_scores)."""
        dev_acc, dev_res = self.evaluate(solver_code, "dev")
        fb = {"task": SOLVER_TASK.format(budget=self.a.solve_budget), "score": dev_acc,
              "failures": self.failures_text(dev_res), "n_candidates": self.a.candidates}
        cands, _ = self.improve(improver_code, solver_code, fb)
        scored = [(self.evaluate(c, "dev")[0], c) for c in cands]
        if not scored:
            return None, -1.0, 0, []
        best_acc, best = max(scored, key=lambda t: t[0])
        return best, best_acc, len(scored), [s for s, _ in scored]

    def improver_utility(self, improver_code, solver_code):
        _, best_acc, _, _ = self.solver_step(improver_code, solver_code)
        return best_acc

    def run(self):
        a = self.a
        out_dir = os.path.join(a.out, f"{a.condition}_seed{a.seed}")
        os.makedirs(os.path.join(out_dir, "code"), exist_ok=True)
        random.seed(a.seed)
        solver, improver = read(os.path.join(HERE, "seeds/solver_seed.py")), read(os.path.join(HERE, "seeds/improver_seed.py"))
        log_path = os.path.join(out_dir, "log.csv")
        fields = ["round", "dev", "test", "accepted", "n_valid", "best_candidate_dev", "cand_scores",
                  "improver_changed", "improver_utility", "solver_len", "improver_len",
                  "solver_flags", "improver_flags", "cum_calls", "cum_in_tokens", "cum_out_tokens", "seconds"]
        fh = open(log_path, "w", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        t0 = time.time()

        def log(rnd, **kw):
            row = {"round": rnd, "dev": round(self.evaluate(solver, "dev")[0], 4),
                   "test": round(self.evaluate(solver, "test")[0], 4),
                   "solver_len": len(solver), "improver_len": len(improver),
                   "solver_flags": int(bool(SUSPICIOUS.search(solver))),
                   "improver_flags": int(bool(SUSPICIOUS.search(improver))),
                   "cum_calls": self.calls, "cum_in_tokens": self.in_tok, "cum_out_tokens": self.out_tok,
                   "seconds": round(time.time() - t0, 1)}
            row.update(kw)
            w.writerow(row)
            fh.flush()
            with open(os.path.join(out_dir, "code", f"solver_r{rnd:02d}.py"), "w", encoding="utf-8") as f:
                f.write(solver)
            with open(os.path.join(out_dir, "code", f"improver_r{rnd:02d}.py"), "w", encoding="utf-8") as f:
                f.write(improver)
            print(f"[{a.condition} s{a.seed}] round {rnd}: dev={row['dev']:.3f} test={row['test']:.3f} "
                  f"acc={kw.get('accepted', '-')} calls={self.calls}", flush=True)

        log(0, accepted="", n_valid="", best_candidate_dev="", cand_scores="",
            improver_changed="", improver_utility="")

        for rnd in range(1, a.rounds + 1):
            cur_dev = self.evaluate(solver, "dev")[0]
            best, best_acc, n_valid, scores = self.solver_step(improver, solver)
            accepted = best is not None and best_acc > cur_dev
            if accepted:
                solver = best

            improver_changed, utility = 0, ""
            if a.condition == "recursive" and rnd % a.meta_every == 0:
                base_util = self.improver_utility(improver, solver)
                fb = {"task": IMPROVER_TASK.format(budget=a.improve_budget), "score": base_util,
                      "failures": f"Current improver produced a best solver with dev accuracy {base_util:.3f}.",
                      "n_candidates": a.candidates}
                cands, _ = self.improve(improver, improver, fb)  # the improver improves itself
                best_util, best_imp = base_util, None
                for c in cands:
                    u = self.improver_utility(c, solver)
                    if u > best_util:
                        best_util, best_imp = u, c
                if best_imp is not None:
                    improver, improver_changed = best_imp, 1
                utility = round(best_util, 4)

            log(rnd, accepted=int(accepted), n_valid=n_valid,
                best_candidate_dev=round(best_acc, 4), cand_scores=";".join(f"{s:.3f}" for s in scores),
                improver_changed=improver_changed, improver_utility=utility)
        fh.close()
        print("done ->", log_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["fixed", "recursive"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--n-dev", type=int, default=50)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--solve-budget", type=int, default=4, help="LLM calls per problem")
    p.add_argument("--improve-budget", type=int, default=6, help="LLM calls per improve() call")
    p.add_argument("--meta-every", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-timeout", type=int, default=3600)
    p.add_argument("--improve-timeout", type=int, default=900)
    p.add_argument("--local-data", default=None, help="optional jsonl instead of HuggingFace GSM8K")
    p.add_argument("--synthetic", action="store_true", help="toy problems (for dry runs)")
    p.add_argument("--out", default="runs")
    Experiment(p.parse_args()).run()


if __name__ == "__main__":
    main()
