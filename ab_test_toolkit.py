"""
A/B Test Design Toolkit
=======================
Experiment design and statistical-methods toolkit for conversion-rate tests.

WHAT THIS IS AND IS NOT:
  A *design and methods* toolkit: sample sizing, minimum detectable effect,
  and Monte Carlo simulations of statistical behaviour. Simulating a method's
  properties is how you show a test design is sound before running it.
  Simulated output is not a business result from a live experiment.

USAGE:
    pip install numpy scipy pandas
    python ab_test_toolkit.py                 # runs all demos
    python ab_test_toolkit.py --baseline 0.032 --mde 0.10 --daily 8000

WHAT IT DOES:
    1. Sample size for a two-proportion test, given baseline, MDE, alpha, power
    2. MDE given a fixed sample size / test duration
    3. Duration planner from daily traffic
    4. Simulation: false-positive rate when you peek at results daily
    5. Simulation: how sequential (alpha-spending) correction fixes it
    6. CUPED variance reduction demo
    7. Sample-ratio-mismatch (SRM) check — the first sanity check on any test
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(42)


# ----------------------------------------------------------------------
# 1. SAMPLE SIZE / MDE
# ----------------------------------------------------------------------
def sample_size_proportions(p1, mde_rel, alpha=0.05, power=0.80, two_sided=True):
    """Sample size PER ARM for detecting a relative lift of mde_rel on p1."""
    p2 = p1 * (1 + mde_rel)
    z_a = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) +
           z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def mde_from_n(p1, n_per_arm, alpha=0.05, power=0.80):
    """Smallest relative lift detectable with n_per_arm, by bisection."""
    lo, hi = 1e-5, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sample_size_proportions(p1, mid, alpha, power) > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi


def duration_table(p1, daily_traffic, alpha=0.05, power=0.80,
                   mdes=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30)):
    rows = []
    for m in mdes:
        n = sample_size_proportions(p1, m, alpha, power)
        total = 2 * n
        rows.append({
            "relative_MDE": f"{m:.0%}",
            "absolute_lift": f"{p1 * m:.4f}",
            "n_per_arm": f"{n:,}",
            "total_n": f"{total:,}",
            "days_needed": round(total / daily_traffic, 1),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. PEEKING SIMULATION
# ----------------------------------------------------------------------
def peeking_simulation(p, n_final, n_looks=10, n_sims=2000, alpha=0.05):
    """
    Under a TRUE NULL (no real effect), how often do we declare significance
    if we test repeatedly as data accumulates vs only once at the end?
    """
    look_points = np.linspace(n_final / n_looks, n_final, n_looks).astype(int)
    false_pos_peeking = 0
    false_pos_fixed = 0

    for _ in range(n_sims):
        a = rng.random(n_final) < p
        b = rng.random(n_final) < p
        hit = False
        for n in look_points:
            ca, cb = a[:n].sum(), b[:n].sum()
            if ca + cb == 0:
                continue
            _, pval = two_prop_ztest(ca, n, cb, n)
            if pval < alpha:
                hit = True
                break
        if hit:
            false_pos_peeking += 1

        _, pval_end = two_prop_ztest(a.sum(), n_final, b.sum(), n_final)
        if pval_end < alpha:
            false_pos_fixed += 1

    return {
        "nominal_alpha": alpha,
        "false_positive_fixed_horizon": false_pos_fixed / n_sims,
        "false_positive_with_peeking": false_pos_peeking / n_sims,
        "inflation_factor": round((false_pos_peeking / n_sims) /
                                  max(false_pos_fixed / n_sims, 1e-9), 2),
        "n_looks": n_looks,
        "n_sims": n_sims,
    }


def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def obrien_fleming_bounds(n_looks, alpha=0.05):
    """Approximate O'Brien-Fleming alpha-spending boundaries."""
    t = np.arange(1, n_looks + 1) / n_looks
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return z_alpha / np.sqrt(t)


def corrected_peeking_simulation(p, n_final, n_looks=10, n_sims=2000, alpha=0.05):
    look_points = np.linspace(n_final / n_looks, n_final, n_looks).astype(int)
    bounds = obrien_fleming_bounds(n_looks, alpha)
    false_pos = 0
    for _ in range(n_sims):
        a = rng.random(n_final) < p
        b = rng.random(n_final) < p
        for i, n in enumerate(look_points):
            z, _ = two_prop_ztest(a[:n].sum(), n, b[:n].sum(), n)
            if abs(z) > bounds[i]:
                false_pos += 1
                break
    return {"false_positive_with_OBF_correction": false_pos / n_sims,
            "boundaries_z": np.round(bounds, 2).tolist()}


# ----------------------------------------------------------------------
# 3. CUPED
# ----------------------------------------------------------------------
def cuped_demo(n=20000, true_lift=0.0, corr=0.7):
    """
    CUPED: use a pre-experiment covariate to strip out known variance.
    Demonstrates variance reduction, not a business result.
    """
    pre_a = rng.normal(100, 20, n)
    pre_b = rng.normal(100, 20, n)
    noise = np.sqrt(1 - corr ** 2)
    post_a = corr * pre_a + noise * rng.normal(100, 20, n)
    post_b = corr * pre_b + noise * rng.normal(100, 20, n) + true_lift

    pre_all = np.concatenate([pre_a, pre_b])
    post_all = np.concatenate([post_a, post_b])
    theta = np.cov(post_all, pre_all)[0, 1] / np.var(pre_all)
    mu_pre = pre_all.mean()

    adj_a = post_a - theta * (pre_a - mu_pre)
    adj_b = post_b - theta * (pre_b - mu_pre)

    var_before = (post_a.var(ddof=1) + post_b.var(ddof=1)) / 2
    var_after = (adj_a.var(ddof=1) + adj_b.var(ddof=1)) / 2

    return {
        "theta": round(float(theta), 4),
        "variance_before": round(float(var_before), 2),
        "variance_after": round(float(var_after), 2),
        "variance_reduction_pct": round(100 * (1 - var_after / var_before), 1),
        "effective_sample_size_multiplier": round(float(var_before / var_after), 2),
    }


# ----------------------------------------------------------------------
# 4. SRM CHECK
# ----------------------------------------------------------------------
def srm_check(n_control, n_treatment, expected_ratio=0.5):
    total = n_control + n_treatment
    expected = np.array([total * expected_ratio, total * (1 - expected_ratio)])
    observed = np.array([n_control, n_treatment])
    chi2, p = stats.chisquare(observed, expected)
    return {
        "observed_split": f"{n_control:,} / {n_treatment:,}",
        "observed_ratio": round(n_control / total, 4),
        "chi2": round(float(chi2), 2),
        "p_value": round(float(p), 5),
        "verdict": "SRM DETECTED — do not trust results, debug assignment"
                   if p < 0.001 else "No SRM — split looks clean",
    }


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=float, default=0.032,
                    help="baseline conversion rate, e.g. 0.032 = 3.2%%")
    ap.add_argument("--mde", type=float, default=0.10, help="relative MDE, e.g. 0.10 = 10%%")
    ap.add_argument("--daily", type=int, default=8000, help="daily sessions per experiment")
    args = ap.parse_args()

    p = args.baseline
    print("=" * 72)
    print("A/B TEST DESIGN TOOLKIT")
    print("=" * 72)
    print(f"  Baseline conversion   {p:.3%}")
    print(f"  Daily traffic         {args.daily:,} sessions")

    print("\n" + "=" * 72)
    print("1. SAMPLE SIZE / DURATION PLANNER  (alpha=0.05, power=0.80)")
    print("=" * 72)
    print(duration_table(p, args.daily).to_string(index=False))

    n = sample_size_proportions(p, args.mde)
    print(f"\n  For a {args.mde:.0%} relative lift: {n:,} per arm, "
          f"{2 * n:,} total, {2 * n / args.daily:.1f} days")

    print("\n" + "=" * 72)
    print("2. MDE GIVEN A FIXED RUN LENGTH")
    print("=" * 72)
    for days in (7, 14, 21, 28):
        n_arm = int(days * args.daily / 2)
        m = mde_from_n(p, n_arm)
        print(f"  {days:>2} days -> {n_arm:>9,} per arm -> smallest detectable lift {m:.1%}")

    print("\n" + "=" * 72)
    print("3. THE PEEKING PROBLEM  (simulated under a TRUE NULL)")
    print("=" * 72)
    n_final = min(sample_size_proportions(p, 0.10), 60000)
    res = peeking_simulation(p, n_final, n_looks=10, n_sims=1500)
    for k, v in res.items():
        print(f"  {k:<32} {v}")
    print("\n  Read this as: checking daily and stopping at the first significant")
    print("  result inflates your false-positive rate well above the nominal 5%.")

    print("\n" + "=" * 72)
    print("4. SEQUENTIAL CORRECTION (O'Brien-Fleming alpha spending)")
    print("=" * 72)
    corr = corrected_peeking_simulation(p, n_final, n_looks=10, n_sims=1500)
    print(f"  false_positive_with_OBF_correction  {corr['false_positive_with_OBF_correction']}")
    print(f"  z-boundaries by look                {corr['boundaries_z']}")
    print("  Early looks demand a much larger z — that is what buys back alpha.")

    print("\n" + "=" * 72)
    print("5. CUPED VARIANCE REDUCTION")
    print("=" * 72)
    for k, v in cuped_demo().items():
        print(f"  {k:<38} {v}")
    print("  Interpretation: same statistical power from fewer users.")

    print("\n" + "=" * 72)
    print("6. SAMPLE RATIO MISMATCH CHECK")
    print("=" * 72)
    print("  Clean split:")
    for k, v in srm_check(50120, 49880).items():
        print(f"    {k:<20} {v}")
    print("  Broken split:")
    for k, v in srm_check(51800, 48200).items():
        print(f"    {k:<20} {v}")


if __name__ == "__main__":
    main()
