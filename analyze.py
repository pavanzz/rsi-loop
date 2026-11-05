"""Aggregate runs/ into figures and a summary table for the paper.

  python analyze.py --runs runs --out figures

Outputs
  fig_dev_test.png     mean +/- sd dev and test accuracy per round, per condition
  fig_gains.png        mean per-round test gain (are gains compounding or shrinking?)
  summary.csv          per-run totals, dev-test gap, early vs late gains, curve-fit comparison
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"fixed": "#2a6fdb", "recursive": "#d9480f"}


def load(runs_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "log.csv"))):
        name = os.path.basename(os.path.dirname(path))
        cond = name.split("_seed")[0]
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        runs.setdefault(cond, []).append({
            "name": name,
            "dev": np.array([float(r["dev"]) for r in rows]),
            "test": np.array([float(r["test"]) for r in rows]),
            "calls": np.array([float(r["cum_calls"]) for r in rows]),
            "flags": sum(int(r["solver_flags"]) for r in rows),
            "accepted": sum(int(r["accepted"] or 0) for r in rows[1:]),
        })
    return runs


def fit_compare(y):
    """Compare a saturating curve y=a-b*exp(-c t) with a linear trend y=a+b t (AIC; lower is better)."""
    t = np.arange(len(y), dtype=float)
    n = len(y)

    def aic(rss, k):
        return n * np.log(max(rss, 1e-12) / n) + 2 * k

    lin = np.polyfit(t, y, 1)
    rss_lin = float(np.sum((np.polyval(lin, t) - y) ** 2))
    try:
        from scipy.optimize import curve_fit
        f = lambda t, a, b, c: a - b * np.exp(-c * t)  # noqa: E731
        p, _ = curve_fit(f, t, y, p0=[y.max(), max(y.max() - y[0], 1e-3), 0.3],
                         bounds=([0, -1, 0], [1, 2, 10]), maxfev=20000)
        rss_sat = float(np.sum((f(t, *p) - y) ** 2))
        return aic(rss_lin, 2), aic(rss_sat, 3), float(p[0])
    except Exception:  # noqa: BLE001
        return aic(rss_lin, 2), float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.runs)
    if not runs:
        raise SystemExit(f"no runs found in {a.runs}/")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, ["dev", "test"]):
        for cond, rs in runs.items():
            L = min(len(r[split]) for r in rs)
            Y = np.stack([r[split][:L] for r in rs])
            m, s = Y.mean(0), Y.std(0)
            x = np.arange(L)
            ax.plot(x, m, color=COLORS.get(cond), lw=2, label=f"{cond} (n={len(rs)})")
            ax.fill_between(x, m - s, m + s, color=COLORS.get(cond), alpha=0.15, lw=0)
        ax.set_title(f"{split} accuracy")
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_dev_test.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(5, 4))
    for cond, rs in runs.items():
        L = min(len(r["test"]) for r in rs)
        G = np.stack([np.diff(r["test"][:L]) for r in rs])
        ax.plot(np.arange(1, L), G.mean(0), marker="o", color=COLORS.get(cond), label=cond)
    ax.axhline(0, color="#888", lw=1)
    ax.set_xlabel("round")
    ax.set_ylabel("test-accuracy gain vs previous round")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_gains.png"), dpi=200)

    with open(os.path.join(a.out, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "condition", "test_start", "test_end", "test_gain", "dev_end",
                    "final_dev_test_gap", "early_gain_test", "late_gain_test", "accepted_rounds",
                    "flagged_rounds", "total_calls", "aic_linear", "aic_saturating", "fitted_ceiling"])
        for cond, rs in runs.items():
            for r in rs:
                t, d = r["test"], r["dev"]
                half = len(t) // 2
                aic_l, aic_s, ceil = fit_compare(t)
                w.writerow([r["name"], cond, t[0], t[-1], round(t[-1] - t[0], 4), d[-1],
                            round(d[-1] - t[-1], 4), round(t[half] - t[0], 4), round(t[-1] - t[half], 4),
                            r["accepted"], r["flags"], int(r["calls"][-1]),
                            round(aic_l, 2), round(aic_s, 2), round(ceil, 4)])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
