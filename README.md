# Compounding or Saturating? — RSI loop experiment

![dry-run](../../actions/workflows/dry-run.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

> **Status: work in progress.** Experiments are running; results and the arXiv paper will be added here.

**The question:** when an AI improves AI, do the gains *compound* or *flatten out*?

An LLM repeatedly rewrites a math-solving program (the **solver**). In the
**recursive** condition it also rewrites the program that does the rewriting
(the **improver**), STOP-style. We measure whether the gains compound or level off,
and how much of the "improvement" is just selection noise (the dev-test gap).

## 1. Setup (Windows, Mac or Linux, Python 3.10+)

```bash
pip install -r requirements.txt
```

> ⚠️ This runs code written by an LLM on your machine. The sandbox blocks access to the
> answers and limits LLM calls, but it is **not** a security sandbox. Run it in a VM,
> Docker container or spare machine if you can.

## 2. Free dry run (no API key, fake LLM)

```bash
# Mac/Linux
RSI_MOCK=1 python run_experiment.py --condition recursive --rounds 4 --synthetic --out runs_mock
# Windows PowerShell
$env:RSI_MOCK="1"; python run_experiment.py --condition recursive --rounds 4 --synthetic --out runs_mock
python analyze.py --runs runs_mock --out figures_mock
```

## 3. Real run

Any OpenAI-compatible API works (OpenAI, OpenRouter, Together, a local server...).

```bash
export RSI_MODEL="<model name>"
export RSI_API_KEY="<your key>"
export RSI_BASE_URL="<optional, e.g. https://openrouter.ai/api/v1>"
```

**Pilot first** (cheap, checks everything works with the real model):

```bash
python run_experiment.py --condition fixed --rounds 3 --n-dev 30 --n-test 60 --out runs_pilot
```

**Full experiment** (3 seeds x 2 conditions):

```bash
for s in 0 1 2; do
  python run_experiment.py --condition fixed     --seed $s --out runs
  python run_experiment.py --condition recursive --seed $s --out runs
done
python analyze.py --runs runs --out figures
```

Defaults: 15 rounds, 50 dev problems (selection), 200 test problems (held out),
3 candidates per round, 4 LLM calls per problem, recursive meta-step every 3 rounds.
A full run is on the order of 10–30k LLM calls per seed, so use a small, cheap model
and check `cum_calls` / token columns in `runs/*/log.csv` after the pilot.

## 4. What you get

- `runs/<condition>_seed<k>/log.csv` — per-round dev/test accuracy, accepted?, candidate
  scores, improver changed?, code length, suspicious-code flags, cumulative calls/tokens
- `runs/.../code/` — every solver and improver version (great for the paper's qualitative section)
- `figures/fig_dev_test.png`, `figures/fig_gains.png`, `figures/summary.csv`

## Files

| file | role |
|---|---|
| `run_experiment.py` | the loop (fixed vs recursive) |
| `sandbox_worker.py` | runs generated code in a separate process, with call budgets and timeouts |
| `llm.py` | OpenAI-compatible wrapper + mock LLM |
| `data.py` | GSM8K loading and scoring (last number in the output) |
| `seeds/` | starting solver and improver |
| `analyze.py` | figures + summary table |
| `paper_outline.md` | paper structure |

## Related work this builds on

STOP (Zelikman et al., 2023) · Gödel Agent (Yin et al., 2024) · Darwin Gödel Machine (Zhang et al., 2025) ·
ADAS (Hu, Lu & Clune, 2024) · OPRO (Yang et al., 2023) · AlphaEvolve (Novikov et al., 2025) · GSM8K (Cobbe et al., 2021)

## License

MIT — see `LICENSE`. Feedback and issues welcome.
