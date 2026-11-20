"""Sequential Testing & Peeking Protection Engine (Milestone 4)."""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


class SequentialEngine:
    """Implements mSPRT and Lan-DeMets Alpha Spending for continuous peeking without Type I error inflation."""

    @classmethod
    def msprt_proportions(
        cls,
        conversions_a: List[int],
        conversions_b: List[int],
        alpha: float = 0.05,
        tau: float = 0.05  # Mixing parameter for normal mixture prior on effect size
    ) -> Dict[str, Any]:
        """Perform mixture Sequential Probability Ratio Test (mSPRT) on sequential observation streams.
        
        Args:
            conversions_a: Cumulative binary conversion indicators for Variant A
            conversions_b: Cumulative binary conversion indicators for Variant B
            alpha: Target Type I error rate (e.g. 0.05)
            tau: Prior variance parameter on lift
        """
        n = min(len(conversions_a), len(conversions_b))
        if n == 0:
            raise ValueError("Data streams cannot be empty.")

        a_stream = np.array(conversions_a[:n], dtype=float)
        b_stream = np.array(conversions_b[:n], dtype=float)

        # Difference process: Y_i = B_i - A_i
        y = b_stream - a_stream
        cum_s = np.cumsum(y)
        var_sample = max(0.01, float(np.var(y, ddof=1)) if n > 1 else 0.25)

        # Stopping boundary: 1 / alpha
        threshold = 1.0 / alpha

        trajectory = []
        stopped_early = False
        stopping_index = None

        # Calculate mSPRT likelihood ratio at each time step
        for i in range(1, n + 1):
            s_i = float(cum_s[i - 1])
            v_i = i * var_sample
            tau_sq = tau ** 2

            # Mixture likelihood ratio for Normal observation increment
            sigma_sq = v_i
            # Lambda_n = sqrt(sigma^2 / (sigma^2 + n*tau^2)) * exp( (tau^2 * S_n^2) / (2 * sigma^2 * (sigma^2 + n*tau^2)) )
            denom = sigma_sq + (i * tau_sq)
            lambda_n = math.sqrt(sigma_sq / denom) * math.exp((tau_sq * (s_i ** 2)) / (2.0 * sigma_sq * denom))

            # Always-valid confidence sequence
            z_crit = math.sqrt(2.0 * (denom / sigma_sq) * math.log(math.sqrt(denom / sigma_sq) / alpha))
            mean_diff = s_i / i
            width = (math.sqrt(denom) / (i * math.sqrt(sigma_sq))) * z_crit if i > 1 else 0.5
            ci_lower = mean_diff - width
            ci_upper = mean_diff + width

            is_significant = lambda_n >= threshold
            if is_significant and not stopped_early:
                stopped_early = True
                stopping_index = i

            trajectory.append({
                "step": i,
                "diff_cumulative": round(s_i, 3),
                "mean_difference": round(mean_diff, 5),
                "likelihood_ratio": round(float(lambda_n), 3),
                "threshold": round(threshold, 2),
                "confidence_lower": round(ci_lower, 5),
                "confidence_upper": round(ci_upper, 5),
                "can_stop_early": bool(is_significant)
            })

        latest = trajectory[-1]
        return {
            "test_type": "mSPRT_proportions",
            "alpha": alpha,
            "threshold": threshold,
            "total_observations_tested": n,
            "stopped_early": stopped_early,
            "stopping_sample_size": stopping_index if stopped_early else n,
            "current_likelihood_ratio": latest["likelihood_ratio"],
            "decision": "REJECT_NULL_EARLY_STOP" if stopped_early else "CONTINUE_TESTING",
            "latest_confidence_sequence": [latest["confidence_lower"], latest["confidence_upper"]],
            "trajectory": trajectory[::max(1, n // 50)]  # Sample 50 steps for charting
        }

    @classmethod
    def alpha_spending_obrien_fleming(
        cls,
        total_planned_sample_size: int,
        interim_sample_sizes: List[int],
        alpha: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Calculate O'Brien-Fleming Alpha Spending boundaries for scheduled interim analyses."""
        results = []
        cum_alpha_spent = 0.0

        for n_k in interim_sample_sizes:
            info_fraction = min(1.0, max(0.01, n_k / total_planned_sample_size))
            # Lan-DeMets O'Brien-Fleming alpha spending function:
            # alpha(t) = 2 - 2 * Phi( Z_{1 - alpha/2} / sqrt(t) )
            z_half = stats.norm.ppf(1.0 - alpha / 2.0)
            spent_up_to_t = 2.0 * (1.0 - stats.norm.cdf(z_half / math.sqrt(info_fraction)))

            incremental_alpha = max(0.0, spent_up_to_t - cum_alpha_spent)
            cum_alpha_spent = spent_up_to_t

            # Boundary z-statistic for this look
            boundary_z = stats.norm.ppf(1.0 - incremental_alpha / 2.0) if incremental_alpha > 0 else 5.0
            p_boundary = 2.0 * (1.0 - stats.norm.cdf(boundary_z))

            results.append({
                "interim_sample_size": n_k,
                "information_fraction": round(info_fraction, 3),
                "cumulative_alpha_spent": round(spent_up_to_t, 6),
                "marginal_alpha": round(incremental_alpha, 6),
                "critical_z_boundary": round(float(boundary_z), 4),
                "p_value_threshold": round(float(p_boundary), 6)
            })

        return results

    @classmethod
    def simulate_peeking_problem(
        cls,
        sample_size: int = 2000,
        peeks: int = 20,
        num_simulations: int = 500,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Demonstrates Type I error inflation under repeated peeking when H0 is true."""
        false_positives_naive = 0
        false_positives_msprt = 0
        peek_points = np.linspace(sample_size // peeks, sample_size, peeks, dtype=int)

        np.random.seed(42)
        threshold_msprt = 1.0 / alpha

        for _ in range(num_simulations):
            # Generate null data: A and B have exact same distribution N(0, 1)
            a = np.random.normal(0, 1, sample_size)
            b = np.random.normal(0, 1, sample_size)

            naive_rejected = False
            msprt_rejected = False

            y = b - a
            cum_s = np.cumsum(y)

            for step in peek_points:
                # Naive t-test
                t_stat, p_val = stats.ttest_ind(b[:step], a[:step], equal_var=True)
                if p_val < alpha:
                    naive_rejected = True

                # mSPRT test
                s_i = cum_s[step - 1]
                v_i = step * 2.0  # Var(B - A) = 2
                denom = v_i + (step * (0.05 ** 2))
                lambda_n = math.sqrt(v_i / denom) * math.exp(((0.05 ** 2) * (s_i ** 2)) / (2.0 * v_i * denom))
                if lambda_n >= threshold_msprt:
                    msprt_rejected = True

            if naive_rejected:
                false_positives_naive += 1
            if msprt_rejected:
                false_positives_msprt += 1

        empirical_type_1_naive = false_positives_naive / num_simulations
        empirical_type_1_msprt = false_positives_msprt / num_simulations

        return {
            "simulations": num_simulations,
            "nominal_alpha": alpha,
            "number_of_peeks": peeks,
            "empirical_false_positive_naive": round(empirical_type_1_naive, 4),
            "empirical_false_positive_msprt": round(empirical_type_1_msprt, 4),
            "inflation_multiplier": round(empirical_type_1_naive / alpha, 2)
        }
