"""
A/B test analysis for retention interventions.

The churn model (src/model.py) tells you WHO is likely to leave. This module
answers the next question a real business actually needs answered: if you DO
something about it -- a retention discount, proactive outreach, a feature
unlock -- does it actually work, and how would you know for sure rather than
just guess from a before/after comparison that could be explained by
something else entirely (seasonality, a concurrent marketing push, regression
to the mean among a self-selected risky group)?

This module does two things:
  1. Simulates a retention-intervention experiment: takes the model's
     highest-risk customers, randomly assigns them to treatment (gets the
     intervention) vs. control (doesn't), and simulates an outcome with a
     specified true treatment effect -- built this way so the "true" effect
     is known and the statistical test's ability to recover it can be
     honestly demonstrated, the same spirit as testing the drift module
     against a feature we deliberately shifted.
  2. Analyzes an experiment's results properly: two-proportion z-test for
     significance, a confidence interval on the effect size (not just a
     p-value -- a p-value alone doesn't tell you if an effect is big enough
     to matter), and a minimum-sample-size calculator so an experiment can
     be sized correctly *before* it runs rather than analyzed with
     insufficient power after the fact and wrongly called "no effect."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats


@dataclass
class ExperimentResult:
    n_control: int
    n_treatment: int
    churn_rate_control: float
    churn_rate_treatment: float
    absolute_effect: float          # control_rate - treatment_rate (positive = treatment reduced churn)
    relative_effect: float          # absolute_effect / control_rate
    z_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    alpha: float


def simulate_retention_experiment(
    high_risk_customers: pd.DataFrame,
    true_treatment_effect: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly assign high-risk customers to treatment/control and simulate a
    churned/not-churned outcome, where `true_treatment_effect` is the actual
    absolute reduction in churn probability the intervention causes (e.g.
    0.08 = the intervention cuts churn probability by 8 percentage points).
    The customers' own model-predicted churn probability is used as their
    baseline (control-arm) churn rate, so higher-risk customers stay
    higher-risk in the simulation -- the intervention shifts that baseline
    down for the treatment arm rather than replacing it with a flat rate.
    """
    rng = np.random.default_rng(seed)
    df = high_risk_customers.copy().reset_index(drop=True)

    df["arm"] = rng.choice(["control", "treatment"], size=len(df), p=[0.5, 0.5])

    base_churn_prob = df["churn_probability"].values
    effective_prob = np.where(
        df["arm"] == "treatment",
        np.clip(base_churn_prob - true_treatment_effect, 0.01, 0.99),
        base_churn_prob,
    )
    df["churned_in_experiment"] = rng.random(len(df)) < effective_prob
    return df


def analyze_experiment(
    df: pd.DataFrame, arm_col: str = "arm", outcome_col: str = "churned_in_experiment",
    alpha: float = 0.05,
) -> ExperimentResult:
    """
    Two-proportion z-test comparing churn rate in control vs. treatment,
    with a Wald confidence interval on the absolute effect size.
    """
    control = df[df[arm_col] == "control"][outcome_col]
    treatment = df[df[arm_col] == "treatment"][outcome_col]

    n_c, n_t = len(control), len(treatment)
    p_c, p_t = control.mean(), treatment.mean()

    # pooled proportion for the z-test's standard error under H0: p_c == p_t
    p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))

    absolute_effect = p_c - p_t  # positive = treatment reduced churn
    z_stat = absolute_effect / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

    # CI on the effect uses the UNpooled standard error (standard for a CI,
    # as opposed to the pooled SE used for the hypothesis test itself)
    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = absolute_effect - z_crit * se_unpooled
    ci_high = absolute_effect + z_crit * se_unpooled

    return ExperimentResult(
        n_control=n_c,
        n_treatment=n_t,
        churn_rate_control=p_c,
        churn_rate_treatment=p_t,
        absolute_effect=absolute_effect,
        relative_effect=(absolute_effect / p_c) if p_c > 0 else np.nan,
        z_stat=z_stat,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=p_value < alpha,
        alpha=alpha,
    )


def required_sample_size(
    baseline_rate: float, minimum_detectable_effect: float,
    alpha: float = 0.05, power: float = 0.80,
) -> int:
    """
    Minimum sample size PER ARM to detect `minimum_detectable_effect` (an
    absolute change in churn rate) with the given significance level and
    power, using the standard two-proportion z-test sample size formula.

    This is what should be run BEFORE an experiment, not after -- sizing an
    experiment after the fact and calling a null result "no effect" when
    the experiment was simply underpowered to detect it is one of the most
    common real mistakes in applied experimentation.
    """
    p1 = baseline_rate
    p2 = baseline_rate - minimum_detectable_effect
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
        z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p1 - p2) ** 2

    n = numerator / denominator
    return int(np.ceil(n))


def summarize_result(result: ExperimentResult) -> str:
    """Plain-language summary of an experiment result, written the way you'd
    actually report it to a non-technical stakeholder."""
    verdict = "a statistically significant" if result.significant else "not a statistically significant"
    direction = "reduced" if result.absolute_effect > 0 else "increased"
    return (
        f"The intervention showed {verdict} effect on churn "
        f"(p = {result.p_value:.4f}, alpha = {result.alpha}). "
        f"Control arm churn rate: {result.churn_rate_control:.1%} (n={result.n_control}). "
        f"Treatment arm churn rate: {result.churn_rate_treatment:.1%} (n={result.n_treatment}). "
        f"The intervention {direction} churn by {abs(result.absolute_effect):.1%} percentage points "
        f"({result.relative_effect:+.1%} relative change), "
        f"95% CI: [{result.ci_low:.1%}, {result.ci_high:.1%}]."
    )
