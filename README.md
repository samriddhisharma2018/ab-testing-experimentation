# A/B Testing & Experimentation

Statistical toolkit for designing and validating conversion-rate experiments:
sample sizing, minimum detectable effect (MDE), the peeking problem, sequential
testing with O'Brien-Fleming alpha spending, CUPED variance reduction, and
sample-ratio-mismatch (SRM) checks.

> **Note:** The false-positive, alpha-spending and CUPED results below come from
> Monte Carlo simulations (fixed seed, fully reproducible). They show how each
> method behaves, not the outcome of a live experiment.

## Results (default run)

Inputs: 3.2% baseline conversion, 10% relative MDE, alpha = 0.05, power = 0.80,
8,000 sessions/day.

| Area | Result |
|---|---|
| Sample size | **49,777 users per arm** (99,554 total), **12.4 days** to detect a 10% relative lift |
| Peeking (10 interim looks, true null) | False-positive rate rises from 4.2% to **18.8%**, a **4.48× inflation** |
| O'Brien-Fleming alpha spending | False-positive rate brought back down to **7.2%** |
| CUPED (pre-period covariate, ρ = 0.7) | Variance cut by **48.6%**, **1.95× effective sample size** |
| SRM check | 50,120 / 49,880 split passes; 51,800 / 48,200 is flagged (p < 0.001) |

Full console output: [`results/sample_output.txt`](results/sample_output.txt)

## What the toolkit does

1. **Sample size for two proportions.** Normal-approximation formula for users per arm, given baseline, relative MDE, alpha and power.
2. **MDE for a fixed run length.** Solves for the smallest detectable lift at 7, 14, 21 and 28 days by bisection.
3. **Duration planner.** Converts the required sample into days from daily traffic, for a range of MDEs.
4. **Peeking simulation.** Simulates A/A tests and compares stopping at the first significant interim look against a single fixed-horizon test.
5. **Sequential correction.** Applies O'Brien-Fleming boundaries (`z_alpha / sqrt(t)`), which demand much stronger evidence at early looks.
6. **CUPED.** Adjusts the outcome by `theta * (X_pre - mean(X_pre))` to strip out variance that existed before the experiment.
7. **SRM check.** Chi-square test of the observed traffic split against the planned split.

## Run it

```bash
pip install -r requirements.txt
python ab_test_toolkit.py                                     # defaults above
python ab_test_toolkit.py --baseline 0.041 --mde 0.10 --daily 8000
```

| Flag | Meaning | Default |
|---|---|---|
| `--baseline` | Baseline conversion rate | `0.032` |
| `--mde` | Relative minimum detectable effect | `0.10` |
| `--daily` | Daily sessions entering the experiment | `8000` |

## Notes on method

- **Sizing.** The smallest lift worth detecting is a product decision. If the
  required duration runs past about four weeks, the effect is too small to test
  reliably and an underpowered test should not be run.
- **Peeking.** Checking results repeatedly and stopping at the first
  significant result inflates false positives well above the nominal 5%.
  Either commit to a fixed horizon or use sequential boundaries from the start.
- **Alpha spending.** The boundaries use the classic `z_alpha / sqrt(t)`
  approximation, which keeps the final-look boundary at 1.96. That is why the
  corrected rate lands at 7.2% rather than exactly 5%. Exact Lan-DeMets
  spending would close the rest of the gap.
- **SRM first.** If assignment is broken, every downstream metric is
  untrustworthy, so check the traffic split before reading results.
- **Guardrails.** Ship decisions should also check metrics that must not get
  worse, such as latency, refunds or unsubscribes.

## Tech

Python, NumPy, SciPy, pandas
