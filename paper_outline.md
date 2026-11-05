# Paper outline

**Working title:** Compounding or Saturating? Measuring Returns in a Self-Improving LLM Loop

**Target:** arXiv, cs.AI (cross-list cs.LG). About 8 pages + appendix, LaTeX.

## Research questions

- **RQ1 – Shape of returns:** Over repeated rounds, do the gains from self-improvement compound,
  stay linear, or saturate? (fit saturating vs linear curves; early vs late gains)
- **RQ2 – Does recursion help?** Does letting the improver rewrite itself (recursive) beat a
  fixed improver at the same LLM-call budget?
- **RQ3 – Real vs apparent gains:** How much of the dev-set improvement survives on held-out
  test data? (dev-test gap = selection noise / overfitting / "winner's curse")
- **RQ4 – Behaviour:** What strategies do the improved programs discover (self-consistency,
  verification, re-asking...)? Any attempts to game the score?

## Sections

1. **Introduction** — RSI moved from theory to lab practice in 2026 (lab reports, startups,
   automated-research benchmarks). The open question is not *whether* AI improves AI but
   *whether gains compound*. We study this in a small, fully reproducible loop.
   Contributions: (1) open, budget-controlled RSI testbed; (2) measurements of returns per
   round; (3) fixed vs recursive comparison; (4) dev-test gap analysis.
2. **Related work**
   - Self-improving code / scaffolds: STOP (Zelikman et al., 2023), Gödel Agent (Yin et al., 2024),
     Darwin Gödel Machine (Zhang et al., 2025), ADAS (Hu, Lu & Clune, 2024)
   - LLMs as optimizers / prompt evolution: OPRO (Yang et al., 2023), Promptbreeder
     (Fernando et al., 2023), DSPy (Khattab et al., 2023)
   - AI for AI research: AlphaEvolve (Novikov et al., 2025), ASI-Evolve (arXiv:2603.29640, 2026),
     Recursive "First Steps Toward Automated AI Research" (2026)
   - RSI discourse and risk: Anthropic "When AI builds itself" (2026); classic intelligence-explosion
     arguments (Good, 1965)
   - Benchmark: GSM8K (Cobbe et al., 2021); reward hacking / Goodhart's law
   *(verify every citation before submission)*
3. **Method**
   - Task, scoring (last number), dev (50, from GSM8K train) / test (200, GSM8K test) split
   - Solver interface; fixed per-problem call budget (prevents "improvement" by just using more compute)
   - Improver interface; fixed per-call budget; elitist acceptance on dev
   - Recursive condition: meta-step every 3 rounds; utility = best dev score of solvers produced
   - Sandbox: separate process, no access to answers, timeouts, static flags
   - Figure 1: loop diagram (solver ← improver ← improver)
4. **Experimental setup** — model(s), 15 rounds, 3 seeds × 2 conditions, total calls/tokens, cost
5. **Results**
   - Fig. 2: dev and test accuracy per round (mean ± sd)
   - Fig. 3: per-round gain (does it shrink?)
   - Table 1: summary.csv (gain, dev-test gap, AIC linear vs saturating, fitted ceiling)
   - Qualitative: examples of discovered strategies from `runs/*/code/`
6. **Discussion** — what this says (and does not say) about RSI in frontier labs; compute limits,
   evaluation as the bottleneck, selection noise as a hidden cause of "fake progress"
7. **Limitations** — one small task, small models/budget, small dev set, few seeds,
   greedy acceptance, single benchmark may be contaminated
8. **Conclusion**

**Appendix:** prompts, seed programs, full logs, best evolved solver/improver code, compute budget.

## Abstract (fill in numbers after the runs)

> Recursive self-improvement (RSI) is increasingly discussed as a near-term possibility, but the
> key empirical question — whether successive self-improvements compound or saturate — is rarely
> measured directly. We build a small, budget-controlled testbed in which a language model
> iteratively rewrites a program that solves grade-school math problems and, in a recursive
> condition, also rewrites the program that performs the rewriting. Across [N] seeds and [R] rounds
> with [MODEL], held-out accuracy rose from [X]% to [Y]%, with per-round gains [shrinking / ...].
> Recursive self-modification [did / did not] outperform a fixed improver at equal budget, and
> [Z] points of the apparent dev-set gain did not transfer to held-out data. We release code and logs.
