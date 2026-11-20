"""Bayesian A/B Testing Engine (Milestone 5)."""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats

try:
    import pymc as pm
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


class BayesianABEngine:
    """Computes exact posterior distributions, P(B > A), Expected Loss, and Credible Intervals."""

    @classmethod
    def analyze_beta_binomial(
        cls,
        conversions_a: int,
        visitors_a: int,
        conversions_b: int,
        visitors_b: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        num_draws: int = 100000,
        rope_threshold: float = 0.001
    ) -> Dict[str, Any]:
        """Analyze binary A/B conversion rates with Beta-Binomial conjugate model.
        
        Args:
            conversions_a, visitors_a: Successes and trials for Control (A)
            conversions_b, visitors_b: Successes and trials for Variant (B)
            prior_alpha, prior_beta: Prior hyperparameters (Beta(1,1) is flat uninformative)
            num_draws: Number of Monte Carlo samples from posterior
            rope_threshold: Region of Practical Equivalence half-width
        """
        # Posterior parameters
        post_alpha_a = prior_alpha + conversions_a
        post_beta_a = prior_beta + (visitors_a - conversions_a)

        post_alpha_b = prior_alpha + conversions_b
        post_beta_b = prior_beta + (visitors_b - conversions_b)

        # Posterior expectations & variances
        mean_a = post_alpha_a / (post_alpha_a + post_beta_a)
        mean_b = post_alpha_b / (post_alpha_b + post_beta_b)

        var_a = (post_alpha_a * post_beta_a) / (((post_alpha_a + post_beta_a) ** 2) * (post_alpha_a + post_beta_a + 1.0))
        var_b = (post_alpha_b * post_beta_b) / (((post_alpha_b + post_beta_b) ** 2) * (post_alpha_b + post_beta_b + 1.0))

        # Sample from joint posterior
        np.random.seed(42)
        theta_a_samples = np.random.beta(post_alpha_a, post_beta_a, size=num_draws)
        theta_b_samples = np.random.beta(post_alpha_b, post_beta_b, size=num_draws)

        # Difference & relative lift
        diff_samples = theta_b_samples - theta_a_samples
        rel_lift_samples = (diff_samples / theta_a_samples) * 100.0

        # Probability B > A
        prob_b_beats_a = float(np.mean(theta_b_samples > theta_a_samples))

        # Expected Loss (in rate units): loss if we choose B but A was actually better
        expected_loss_b = float(np.mean(np.maximum(0.0, theta_a_samples - theta_b_samples)))
        expected_loss_a = float(np.mean(np.maximum(0.0, theta_b_samples - theta_a_samples)))

        # 95% Highest Density Interval (HDI)
        hdi_lower = float(np.percentile(diff_samples, 2.5))
        hdi_upper = float(np.percentile(diff_samples, 97.5))

        rel_hdi_lower = float(np.percentile(rel_lift_samples, 2.5))
        rel_hdi_upper = float(np.percentile(rel_lift_samples, 97.5))

        # ROPE (Region of Practical Equivalence [-rope, +rope])
        prob_in_rope = float(np.mean(np.abs(diff_samples) <= rope_threshold))

        # Decision rule
        if prob_b_beats_a >= 0.95 and expected_loss_b < 0.001:
            decision = "DEPLOY_B"
            decision_msg = "Strong evidence variant B outperforms control with minimal expected loss."
        elif prob_b_beats_a <= 0.05 and expected_loss_a < 0.001:
            decision = "DEPLOY_A"
            decision_msg = "Strong evidence control outperforms variant B; do not ship."
        elif prob_in_rope > 0.90:
            decision = "EQUIVALENT"
            decision_msg = "Posterior mass is concentrated inside ROPE; variants are practically identical."
        else:
            decision = "INCONCLUSIVE"
            decision_msg = "More data required to reach definitive decision threshold."

        # Density curve coordinates for charting
        x_grid = np.linspace(max(0.0, min(mean_a, mean_b) - 4 * math.sqrt(max(var_a, var_b))),
                             min(1.0, max(mean_a, mean_b) + 4 * math.sqrt(max(var_a, var_b))), 60)
        pdf_a = stats.beta.pdf(x_grid, post_alpha_a, post_beta_a)
        pdf_b = stats.beta.pdf(x_grid, post_alpha_b, post_beta_b)

        return {
            "model": "beta_binomial_conjugate",
            "control": {
                "visitors": visitors_a,
                "conversions": conversions_a,
                "posterior_alpha": post_alpha_a,
                "posterior_beta": post_beta_a,
                "posterior_mean": round(float(mean_a), 5),
                "posterior_std": round(math.sqrt(var_a), 5)
            },
            "variant": {
                "visitors": visitors_b,
                "conversions": conversions_b,
                "posterior_alpha": post_alpha_b,
                "posterior_beta": post_beta_b,
                "posterior_mean": round(float(mean_b), 5),
                "posterior_std": round(math.sqrt(var_b), 5)
            },
            "probability_b_beats_a": round(prob_b_beats_a, 4),
            "probability_a_beats_b": round(1.0 - prob_b_beats_a, 4),
            "expected_loss_deploy_b": round(expected_loss_b, 6),
            "expected_loss_deploy_a": round(expected_loss_a, 6),
            "absolute_lift_95_hdi": [round(hdi_lower, 5), round(hdi_upper, 5)],
            "relative_lift_pct_95_hdi": [round(rel_hdi_lower, 2), round(rel_hdi_upper, 2)],
            "prob_in_rope": round(prob_in_rope, 4),
            "decision": decision,
            "decision_summary": decision_msg,
            "pdf_curve": {
                "x": [round(float(v), 5) for v in x_grid],
                "pdf_control": [round(float(v), 4) for v in pdf_a],
                "pdf_variant": [round(float(v), 4) for v in pdf_b]
            }
        }

    @classmethod
    def analyze_continuous_normal(
        cls,
        data_a: List[float],
        data_b: List[float],
        num_draws: int = 50000
    ) -> Dict[str, Any]:
        """Bayesian estimation for two independent groups on continuous metrics."""
        a = np.array(data_a, dtype=float)
        b = np.array(data_b, dtype=float)

        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        se_a = float(stats.sem(a))
        se_b = float(stats.sem(b))

        # Normal posterior approximations for sample means
        np.random.seed(42)
        samples_a = np.random.normal(mean_a, se_a, size=num_draws)
        samples_b = np.random.normal(mean_b, se_b, size=num_draws)

        diff = samples_b - samples_a
        prob_b_greater = float(np.mean(diff > 0))
        exp_loss_b = float(np.mean(np.maximum(0.0, -diff)))

        hdi_lower = float(np.percentile(diff, 2.5))
        hdi_upper = float(np.percentile(diff, 97.5))

        return {
            "model": "bayesian_normal_means",
            "control_mean": round(mean_a, 4),
            "variant_mean": round(mean_b, 4),
            "probability_b_beats_a": round(prob_b_greater, 4),
            "expected_loss_deploy_b": round(exp_loss_b, 4),
            "difference_mean": round(mean_b - mean_a, 4),
            "difference_95_hdi": [round(hdi_lower, 4), round(hdi_upper, 4)]
        }
