"""Summarise the search-budget and surrogate experiment produced by step 13.

Reads the deposited results/search_budget_surrogate_experiment.csv (or a fresh
run of step 13 in results/regenerated/, which takes precedence) and prints the
two contrasts it was built to answer, with the statistics quoted in the article:

  budget    bayes15 against random5  -- is more search effort worth anything?
  surrogate bayes15 against random15 -- at a fixed budget, does the Gaussian
                                        process beat plain random draws?

Both are reported as paired comparisons over the 40 algorithm-seed pairs, using
a Wilcoxon signed-rank test on the non-zero differences and a sign test on their
direction. Held-out performance is an outcome here, never a selection criterion.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest, wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.environ.get("OUT_DIR", REPO_ROOT / "results" / "regenerated"))

FILENAME = "search_budget_surrogate_experiment.csv"
path = OUT_DIR / FILENAME
if not path.exists():
    path = REPO_ROOT / "results" / FILENAME
print(f"reading {path}\n")

data = pd.read_csv(path)
wide = data.pivot_table(index=["seed", "algorithm"], columns="condition",
                        values=["search_kappa", "heldout_kappa", "heldout_auc"])


def describe(label, internal_gain, heldout_delta):
    worse = heldout_delta[heldout_delta < -1e-9]
    better = heldout_delta[heldout_delta > 1e-9]
    nonzero = heldout_delta[np.abs(heldout_delta) > 1e-9]
    print(f"=== {label}  (n = {len(internal_gain)} algorithm-seed pairs)")
    print(f"  internal search kappa : median {np.median(internal_gain):+.4f}"
          f"   mean {internal_gain.mean():+.4f}")
    print(f"  held-out kappa        : median {np.median(heldout_delta):+.4f}"
          f"   mean {heldout_delta.mean():+.4f}")
    print(f"  held-out worse in {len(worse)} (mean {worse.mean() if len(worse) else 0:+.4f}),"
          f" better in {len(better)} (mean {better.mean() if len(better) else 0:+.4f}),"
          f" unchanged in {len(heldout_delta) - len(worse) - len(better)}")
    if len(nonzero) >= 6:
        print(f"  Wilcoxon signed-rank on the held-out change: p = {wilcoxon(nonzero).pvalue:.4f}"
              f"  (n = {len(nonzero)})")
        # For a null result the interval is what carries the information: it bounds
        # how large a real effect could be and still have gone unseen here.
        se = stats.sem(nonzero)
        low, high = stats.t.interval(0.95, len(nonzero) - 1, loc=nonzero.mean(), scale=se)
        mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * nonzero.std(ddof=1) / np.sqrt(len(nonzero))
        print(f"  95% CI of the mean held-out change: [{low:+.4f}, {high:+.4f}]")
        print(f"  smallest difference detectable at 80% power: {mde:.4f} kappa"
              f"  (one compound of the 96-ligand held-out set is worth {1/96:.4f})")
    if len(worse) + len(better) >= 1:
        p = binomtest(len(worse), len(worse) + len(better), 0.5).pvalue
        print(f"  sign test, worse against better: {len(worse)}/{len(worse) + len(better)}, p = {p:.4f}")
    print()


gain_budget = (wide[("search_kappa", "bayes15")] - wide[("search_kappa", "random5")]).to_numpy()
delta_budget = (wide[("heldout_kappa", "bayes15")] - wide[("heldout_kappa", "random5")]).to_numpy()
improved = gain_budget > 1e-9
describe("BUDGET: 15 evaluations against the retained 5, where the internal score improved",
         gain_budget[improved], delta_budget[improved])

bayes, rand = wide[("search_kappa", "bayes15")].to_numpy(), wide[("search_kappa", "random15")].to_numpy()
hb, hr = wide[("heldout_kappa", "bayes15")].to_numpy(), wide[("heldout_kappa", "random15")].to_numpy()
changed = ~np.isclose(bayes, rand, atol=1e-9)
print(f"=== SURROGATE: Gaussian process against random draws at the same budget of 15")
print(f"  the surrogate changed the selected champion in {changed.sum()} of {len(bayes)} runs"
      f" ({100 * changed.sum() / len(bayes):.0f}%)")
print(f"  where it changed, its internal score was higher in {(bayes[changed] > rand[changed]).sum()}"
      f" and lower in {(bayes[changed] < rand[changed]).sum()}")
describe("  held-out consequence of those changes", (bayes - rand)[changed], (hb - hr)[changed])

print("=== mean internal search kappa by condition")
print(data.groupby("condition")["search_kappa"].mean().round(4).to_string())
print("\n=== surrogates actually fitted, by condition")
print(data.groupby("condition")["surrogates_fitted"].agg(["min", "max"]).to_string())
print("\nA value of 0 or 1 means the Gaussian process was never used to propose a point:\n"
      "every evaluated configuration came from the optimizer's initialization draws.")
