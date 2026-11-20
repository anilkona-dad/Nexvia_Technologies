"""Unit and Integration Tests for Causal Inference & Experimentation Platform."""

import numpy as np
import pandas as pd

from backend.app.core.power_calc import PowerCalculator
from backend.app.core.frequentist_tests import FrequentistEngine
from backend.app.core.sequential_tests import SequentialEngine
from backend.app.core.bayesian_ab import BayesianABEngine
from backend.app.core.causal_observational import ObservationalCausalEngine
from backend.app.core.plain_english import PlainEnglishExplainer
from backend.app.data_loader import BenchmarkDataLoader


def test_power_calculator():
    res = PowerCalculator.calculate_sample_size_proportion(
        baseline_rate=0.10,
        mde_relative=0.05,
        alpha=0.05,
        power=0.80
    )
    assert res["sample_size_control"] > 10000
    assert res["total_sample_size"] == res["sample_size_control"] * 2

    # Power curve
    curve = PowerCalculator.generate_power_curve(baseline_rate=0.10, mde_relative=0.05)
    assert len(curve) > 0
    assert curve[-1]["statistical_power"] > curve[0]["statistical_power"]


def test_frequentist_tests():
    # 1. Welch t-test
    control = [10.2, 9.8, 10.5, 11.0, 9.9, 10.1]
    variant = [12.5, 13.0, 12.2, 11.9, 12.8, 13.1]
    t_res = FrequentistEngine.two_sample_t_test(control, variant)
    assert t_res["is_significant"] is True
    assert t_res["p_value"] < 0.01

    # 2. Chi-square test
    chi_res = FrequentistEngine.chi_square_test(
        conversions_control=100, visitors_control=1000,
        conversions_variant=150, visitors_variant=1000
    )
    assert chi_res["is_significant"] is True
    assert chi_res["difference"]["relative_lift_pct"] == 50.0

    # 3. Multiple Testing Correction
    tests = [{"p_value": 0.01}, {"p_value": 0.04}, {"p_value": 0.20}]
    corrected = FrequentistEngine.apply_multiple_testing_correction(tests, method="benjamini_hochberg")
    assert len(corrected) == 3
    assert "adjusted_p_value" in corrected[0]


def test_sequential_tests():
    # mSPRT on streaming data
    np.random.seed(42)
    a = np.random.binomial(1, 0.10, 500).tolist()
    b = np.random.binomial(1, 0.16, 500).tolist()

    seq_res = SequentialEngine.msprt_proportions(a, b, alpha=0.05)
    assert "current_likelihood_ratio" in seq_res
    assert seq_res["threshold"] == 20.0
    assert len(seq_res["trajectory"]) > 0

    # Alpha spending
    interim = [500, 1000, 1500, 2000]
    obf = SequentialEngine.alpha_spending_obrien_fleming(2000, interim, alpha=0.05)
    assert len(obf) == 4
    assert obf[-1]["cumulative_alpha_spent"] <= 0.055


def test_bayesian_ab():
    # Beta-Binomial conjugate test
    bayes_res = BayesianABEngine.analyze_beta_binomial(
        conversions_a=100, visitors_a=1000,
        conversions_b=150, visitors_b=1000
    )
    assert bayes_res["probability_b_beats_a"] > 0.99
    assert bayes_res["expected_loss_deploy_b"] < 0.001
    assert len(bayes_res["pdf_curve"]["x"]) > 0


def test_causal_observational_psm():
    df = BenchmarkDataLoader.load_lalonde_psm_data(n_samples=400)
    psm_res = ObservationalCausalEngine.propensity_score_matching(
        df=df,
        treatment_col="treat",
        outcome_col="re78_outcome",
        covariate_cols=["age", "education", "black", "hispanic", "married", "nodegree"],
        caliper=0.15
    )
    assert psm_res["matched_pairs"] > 0
    assert "att" in psm_res
    assert "covariate_balance" in psm_res


def test_causal_observational_did():
    df = BenchmarkDataLoader.load_card_krueger_did_data()
    did_res = ObservationalCausalEngine.difference_in_differences(
        df=df,
        outcome_col="fte_employment",
        group_col="treated_state_nj",
        time_col="post_period"
    )
    assert "did_treatment_effect" in did_res
    assert did_res["did_treatment_effect"] > 0, "NJ minimum wage policy had positive relative FTE employment effect"


def test_causal_observational_scm():
    df = BenchmarkDataLoader.load_california_prop99_scm_data()
    scm_res = ObservationalCausalEngine.synthetic_control(
        panel_df=df,
        unit_col="state",
        time_col="year",
        outcome_col="cigsale",
        treated_unit_id="California",
        intervention_time=1988
    )
    assert scm_res["average_post_treatment_effect"] < 0, "Prop 99 reduced cigarette consumption relative to synthetic twin"
    assert scm_res["pre_intervention_rmse"] < 10.0


def test_plain_english_explainer():
    res = {
        "p_value": 0.012,
        "is_significant": True,
        "difference": {"relative_lift_pct": 5.4, "ci_lower": 1.2, "ci_upper": 9.6}
    }
    exp = PlainEnglishExplainer.explain("welch_t_test", res)
    assert exp["verdict"] == "WINNER_DETECTED"
    assert "Safe to ship" in exp["recommendation"]
