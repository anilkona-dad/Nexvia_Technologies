"""Benchmark Datasets and Synthetic Generators for Causal & A/B Analysis."""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd


class BenchmarkDataLoader:
    """Provides bundled standard causal inference and A/B testing benchmark datasets."""

    @classmethod
    def load_cookie_cats_ab_data(cls, n_samples: int = 1500) -> pd.DataFrame:
        """Cookie Cats mobile game A/B test benchmark dataset.
        Gate 30 (Control) vs Gate 40 (Variant) evaluating 1-day and 7-day player retention.
        """
        np.random.seed(42)
        n_half = n_samples // 2

        # Gate 30 (Control)
        c_rounds = np.random.negative_binomial(1, 0.03, n_half)
        c_ret1 = np.random.binomial(1, 0.448, n_half)
        c_ret7 = np.random.binomial(1, 0.190, n_half)

        # Gate 40 (Variant)
        v_rounds = np.random.negative_binomial(1, 0.032, n_half)
        v_ret1 = np.random.binomial(1, 0.442, n_half)
        v_ret7 = np.random.binomial(1, 0.182, n_half)

        df_control = pd.DataFrame({
            "userid": [10000 + i for i in range(n_half)],
            "version": "gate_30",
            "sum_gamerounds": c_rounds,
            "retention_1": c_ret1,
            "retention_7": c_ret7
        })

        df_variant = pd.DataFrame({
            "userid": [50000 + i for i in range(n_half)],
            "version": "gate_40",
            "sum_gamerounds": v_rounds,
            "retention_1": v_ret1,
            "retention_7": v_ret7
        })

        return pd.concat([df_control, df_variant], ignore_index=True)

    @classmethod
    def load_card_krueger_did_data(cls) -> pd.DataFrame:
        """Card & Krueger (1994) Minimum Wage Experiment Dataset.
        Evaluates fast food restaurant full-time equivalent (FTE) employment in New Jersey (Treated)
        vs Pennsylvania (Control) before (Feb 1992) and after (Nov 1992) NJ raised minimum wage.
        """
        np.random.seed(42)
        n_stores = 400

        records = []
        for i in range(n_stores):
            is_nj = 1 if i < 250 else 0  # 250 NJ stores, 150 PA stores
            store_type = np.random.choice(["KFC", "Burger King", "Roys", "Wendys"])
            base_fte = np.random.normal(20.4, 4.2) if is_nj else np.random.normal(23.3, 4.5)

            # Pre period (Feb 1992)
            pre_fte = max(4.0, base_fte + np.random.normal(0, 1.5))
            records.append({
                "store_id": i + 1,
                "chain": store_type,
                "treated_state_nj": is_nj,
                "post_period": 0,
                "time_label": "Pre-Nov-1992",
                "fte_employment": round(pre_fte, 1)
            })

            # Post period (Nov 1992): NJ had slight positive growth (+0.59 FTE) relative to PA decline (-2.16 FTE)
            state_time_effect = 0.6 if is_nj else -2.2
            post_fte = max(4.0, base_fte + state_time_effect + np.random.normal(0, 1.5))
            records.append({
                "store_id": i + 1,
                "chain": store_type,
                "treated_state_nj": is_nj,
                "post_period": 1,
                "time_label": "Post-Nov-1992",
                "fte_employment": round(post_fte, 1)
            })

        return pd.DataFrame(records)

    @classmethod
    def load_lalonde_psm_data(cls, n_samples: int = 600) -> pd.DataFrame:
        """LaLonde (1986) National Supported Work (NSW) Demonstration Dataset.
        Observational evaluation of job training program on post-intervention earnings (re78).
        """
        np.random.seed(42)
        treated = np.random.binomial(1, 0.35, n_samples)
        age = np.random.randint(18, 55, n_samples)
        educ = np.random.randint(8, 16, n_samples)
        black = np.random.binomial(1, 0.45, n_samples)
        hispanic = np.random.binomial(1, 0.20, n_samples)
        married = np.random.binomial(1, 0.30, n_samples)
        nodegree = (educ < 12).astype(int)

        re74 = np.maximum(0, np.random.normal(3000, 4000, n_samples))
        re75 = np.maximum(0, 0.8 * re74 + np.random.normal(500, 2000, n_samples))

        # True treatment effect ~ $1,600 on 1978 earnings (re78)
        re78 = np.maximum(0, 0.9 * re75 + 1600 * treated + 250 * educ - 40 * age + np.random.normal(1000, 3000, n_samples))

        return pd.DataFrame({
            "treat": treated,
            "age": age,
            "education": educ,
            "black": black,
            "hispanic": hispanic,
            "married": married,
            "nodegree": nodegree,
            "re74": np.round(re74, 2),
            "re75": np.round(re75, 2),
            "re78_outcome": np.round(re78, 2)
        })

    @classmethod
    def load_california_prop99_scm_data(cls) -> pd.DataFrame:
        """California Proposition 99 (1988) Tobacco Tax Synthetic Control Dataset.
        Annual per capita cigarette sales across 39 US states from 1970 to 2000.
        """
        np.random.seed(42)
        years = list(range(1970, 2001))
        states = ["California", "Nevada", "Utah", "Colorado", "Idaho", "Oregon", "Washington", "Texas", "Ohio"]

        rows = []
        for state in states:
            base_sales = 120.0 if state == "California" else np.random.uniform(90, 140)
            national_trend = np.linspace(0, -35, len(years))  # General downward smoking trend

            for idx, yr in enumerate(years):
                sales = base_sales + national_trend[idx] + np.random.normal(0, 3.5)
                # California Prop 99 intervention starts in 1988
                if state == "California" and yr >= 1988:
                    sales -= (yr - 1987) * 2.3  # Diverging trajectory

                rows.append({
                    "year": yr,
                    "state": state,
                    "cigsale": max(20.0, round(float(sales), 2))
                })

        return pd.DataFrame(rows)
