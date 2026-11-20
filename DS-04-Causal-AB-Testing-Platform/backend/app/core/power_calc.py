"""Power and Sample Size Calculator Engine (Milestone 2)."""

import math
from typing import Any, Dict, List, Optional
import numpy as np

try:
    from statsmodels.stats.power import NormalIndPower, TTestIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class PowerCalculator:
    """Calculates statistical power, required sample sizes, and MDE sensitivity curves."""

    @classmethod
    def calculate_sample_size_proportion(
        cls,
        baseline_rate: float,
        mde_relative: float,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0  # n2 / n1
    ) -> Dict[str, Any]:
        """Calculate required sample size per variant for binary metrics (e.g., conversion rate).
        
        Args:
            baseline_rate: Baseline conversion rate (e.g. 0.10 for 10%)
            mde_relative: Minimum Detectable Effect relative to baseline (e.g. 0.05 for +5% relative)
            alpha: Significance level (default 0.05)
            power: Statistical power 1 - beta (default 0.80)
            ratio: Ratio of variant sample size to control sample size (default 1.0)
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1.0 + mde_relative)
        p2 = max(0.0001, min(0.9999, p2))
        abs_diff = abs(p2 - p1)

        if abs_diff == 0:
            raise ValueError("MDE cannot be zero.")

        # Standard normal quantiles
        from scipy.stats import norm
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        z_beta = norm.ppf(power)

        # Pooled variance under H0, unpooled under H1
        p_bar = (p1 + ratio * p2) / (1.0 + ratio)
        var_h0 = p_bar * (1.0 - p_bar) * (1.0 + 1.0 / ratio)
        var_h1 = (p1 * (1.0 - p1)) + (p2 * (1.0 - p2) / ratio)

        n1 = math.ceil(((z_alpha * math.sqrt(var_h0) + z_beta * math.sqrt(var_h1)) ** 2) / (abs_diff ** 2))
        n2 = math.ceil(n1 * ratio)

        return {
            "metric_type": "proportion",
            "baseline_rate": baseline_rate,
            "target_rate": round(p2, 5),
            "mde_relative": mde_relative,
            "mde_absolute": round(abs_diff, 5),
            "alpha": alpha,
            "power": power,
            "sample_size_control": n1,
            "sample_size_variant": n2,
            "total_sample_size": n1 + n2
        }

    @classmethod
    def calculate_sample_size_continuous(
        cls,
        mean: float,
        std_dev: float,
        mde_relative: float,
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0
    ) -> Dict[str, Any]:
        """Calculate required sample size per variant for continuous metrics (e.g., revenue, session duration)."""
        delta = abs(mean * mde_relative)
        if delta == 0 or std_dev <= 0:
            raise ValueError("Effect size and standard deviation must be positive.")

        from scipy.stats import norm
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        z_beta = norm.ppf(power)

        # Two-sample continuous t-test sample size formula
        cohen_d = delta / std_dev
        n1 = math.ceil(2.0 * ((z_alpha + z_beta) ** 2) * (std_dev ** 2) / (delta ** 2))
        n2 = math.ceil(n1 * ratio)

        return {
            "metric_type": "continuous",
            "baseline_mean": mean,
            "std_dev": std_dev,
            "target_mean": round(mean * (1.0 + mde_relative), 4),
            "cohen_d": round(cohen_d, 4),
            "mde_relative": mde_relative,
            "mde_absolute": round(delta, 4),
            "alpha": alpha,
            "power": power,
            "sample_size_control": n1,
            "sample_size_variant": n2,
            "total_sample_size": n1 + n2
        }

    @classmethod
    def generate_power_curve(
        cls,
        baseline_rate: float,
        sample_sizes: Optional[List[int]] = None,
        alpha: float = 0.05,
        mde_relative: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Generate power sensitivity curve over a grid of sample sizes."""
        from scipy.stats import norm

        if sample_sizes is None:
            sample_sizes = [500, 1000, 2500, 5000, 10000, 20000, 50000]

        p1 = baseline_rate
        p2 = baseline_rate * (1.0 + mde_relative)
        abs_diff = abs(p2 - p1)
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        p_bar = (p1 + p2) / 2.0
        se = math.sqrt(2.0 * p_bar * (1.0 - p_bar))

        curve = []
        for n in sample_sizes:
            se_n = se / math.sqrt(n)
            z_score = (abs_diff / se_n) - z_alpha
            power = norm.cdf(z_score)
            curve.append({
                "sample_size_per_variant": n,
                "total_sample_size": n * 2,
                "statistical_power": round(float(power), 4),
                "mde_relative": mde_relative
            })
        return curve

    @classmethod
    def calculate_mde(
        cls,
        baseline_rate: float,
        sample_size_per_variant: int,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> Dict[str, Any]:
        """Calculate Minimum Detectable Effect (MDE) given a fixed traffic/sample size."""
        from scipy.stats import norm
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        z_beta = norm.ppf(power)

        p = baseline_rate
        se_factor = math.sqrt(2.0 * p * (1.0 - p) / sample_size_per_variant)
        mde_abs = (z_alpha + z_beta) * se_factor
        mde_rel = mde_abs / p

        return {
            "sample_size_per_variant": sample_size_per_variant,
            "baseline_rate": baseline_rate,
            "alpha": alpha,
            "power": power,
            "mde_absolute": round(float(mde_abs), 5),
            "mde_relative": round(float(mde_rel), 5),
            "target_rate": round(float(p + mde_abs), 5)
        }
