"""Frequentist Hypothesis Testing and Multiple Testing Corrections Engine (Milestone 3)."""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


class FrequentistEngine:
    """Performs parametric and non-parametric hypothesis tests with multiple-testing adjustments."""

    @classmethod
    def two_sample_t_test(
        cls,
        control_data: List[float],
        variant_data: List[float],
        alpha: float = 0.05,
        equal_var: bool = False  # Welch's t-test by default
    ) -> Dict[str, Any]:
        """Perform two-sample Welch's t-test for continuous outcomes."""
        a = np.array(control_data, dtype=float)
        b = np.array(variant_data, dtype=float)

        n_a, n_b = len(a), len(b)
        if n_a < 2 or n_b < 2:
            raise ValueError("Both variants must have at least 2 observations.")

        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
        std_a, std_b = math.sqrt(var_a), math.sqrt(var_b)

        # Welch's t-statistic
        se_diff = math.sqrt((var_a / n_a) + (var_b / n_b))
        if se_diff == 0:
            diff = mean_b - mean_a
            t_stat = 0.0 if diff == 0 else float("inf")
            p_val = 1.0 if diff == 0 else 0.0
            dof = n_a + n_b - 2
        else:
            t_stat, p_val = stats.ttest_ind(b, a, equal_var=equal_var)
            # Satterthwaite degrees of freedom
            dof_num = ((var_a / n_a) + (var_b / n_b)) ** 2
            dof_den = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
            dof = dof_num / dof_den if dof_den > 0 else (n_a + n_b - 2)

        # 95% Confidence Interval for difference
        abs_lift = mean_b - mean_a
        rel_lift = (abs_lift / mean_a) * 100.0 if mean_a != 0 else 0.0
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=dof)
        ci_lower = abs_lift - t_crit * se_diff
        ci_upper = abs_lift + t_crit * se_diff

        return {
            "test_type": "welch_t_test" if not equal_var else "student_t_test",
            "control": {"n": n_a, "mean": round(mean_a, 4), "std": round(std_a, 4)},
            "variant": {"n": n_b, "mean": round(mean_b, 4), "std": round(std_b, 4)},
            "difference": {
                "absolute_lift": round(abs_lift, 4),
                "relative_lift_pct": round(rel_lift, 2),
                "standard_error": round(se_diff, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "confidence_level": 1.0 - alpha
            },
            "t_statistic": round(float(t_stat), 4),
            "degrees_of_freedom": round(float(dof), 2),
            "p_value": float(p_val),
            "is_significant": bool(p_val < alpha),
            "alpha": alpha
        }

    @classmethod
    def chi_square_test(
        cls,
        conversions_control: int,
        visitors_control: int,
        conversions_variant: int,
        visitors_variant: int,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Perform Chi-Square test of independence (2x2 contingency table) and two-proportion z-test."""
        c_conv = conversions_control
        c_non_conv = visitors_control - conversions_control
        v_conv = conversions_variant
        v_non_conv = visitors_variant - conversions_variant

        if min(visitors_control, visitors_variant) <= 0:
            raise ValueError("Visitor counts must be greater than zero.")

        rate_c = c_conv / visitors_control
        rate_v = v_conv / visitors_variant
        abs_diff = rate_v - rate_c
        rel_diff = (abs_diff / rate_c) * 100.0 if rate_c > 0 else 0.0

        # 2x2 Contingency Table
        table = [[c_conv, c_non_conv], [v_conv, v_non_conv]]
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(table, correction=True)

        # Standard Error for Difference of Two Independent Proportions
        se_unpooled = math.sqrt((rate_c * (1.0 - rate_c) / visitors_control) + (rate_v * (1.0 - rate_v) / visitors_variant))
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        ci_lower = abs_diff - z_crit * se_unpooled
        ci_upper = abs_diff + z_crit * se_unpooled

        return {
            "test_type": "chi_square_proportions",
            "control": {
                "visitors": visitors_control,
                "conversions": c_conv,
                "conversion_rate": round(rate_c, 5)
            },
            "variant": {
                "visitors": visitors_variant,
                "conversions": v_conv,
                "conversion_rate": round(rate_v, 5)
            },
            "difference": {
                "absolute_diff": round(abs_diff, 5),
                "relative_lift_pct": round(rel_diff, 2),
                "ci_lower": round(ci_lower, 5),
                "ci_upper": round(ci_upper, 5),
                "confidence_level": 1.0 - alpha
            },
            "chi2_statistic": round(float(chi2_stat), 4),
            "p_value": float(p_val),
            "is_significant": bool(p_val < alpha),
            "alpha": alpha
        }

    @classmethod
    def apply_multiple_testing_correction(
        cls,
        test_results: List[Dict[str, Any]],
        method: str = "benjamini_hochberg",
        q_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Apply Bonferroni (FWER) or Benjamini-Hochberg (FDR) corrections to multiple hypotheses."""
        m = len(test_results)
        if m == 0:
            return []

        # Sort indices by original p-value
        indexed_p = [(i, res.get("p_value", 1.0)) for i, res in enumerate(test_results)]
        sorted_p = sorted(indexed_p, key=lambda x: x[1])

        corrected = [dict(r) for r in test_results]

        if method == "bonferroni":
            alpha_adj = q_threshold / m
            for original_idx, p in sorted_p:
                adj_p = min(1.0, p * m)
                corrected[original_idx]["adjusted_p_value"] = round(adj_p, 6)
                corrected[original_idx]["is_significant_corrected"] = bool(p < alpha_adj)
                corrected[original_idx]["correction_method"] = "bonferroni"

        elif method == "benjamini_hochberg":
            # Rank from 1 to m
            # Find largest k such that P_(k) <= (k/m) * q
            max_significant_rank = -1
            for rank, (original_idx, p) in enumerate(sorted_p, start=1):
                critical_value = (rank / m) * q_threshold
                if p <= critical_value:
                    max_significant_rank = rank

            # Calculate adjusted p-values backwards
            cum_min = 1.0
            adj_p_map = {}
            for rank in range(m, 0, -1):
                orig_idx, p = sorted_p[rank - 1]
                val = min(1.0, p * m / rank)
                cum_min = min(cum_min, val)
                adj_p_map[orig_idx] = cum_min

            for orig_idx, res in enumerate(corrected):
                adj_p = adj_p_map[orig_idx]
                res["adjusted_p_value"] = round(adj_p, 6)
                res["is_significant_corrected"] = bool(adj_p <= q_threshold)
                res["correction_method"] = "benjamini_hochberg"

        return corrected
