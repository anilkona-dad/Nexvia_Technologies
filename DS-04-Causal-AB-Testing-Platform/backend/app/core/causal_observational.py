"""Observational Causal Inference Engine: PSM, Difference-in-Differences, and Synthetic Control (Milestone 6)."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

try:
    import dowhy
    from dowhy import CausalModel
    HAS_DOWHY = True
except ImportError:
    HAS_DOWHY = False


class ObservationalCausalEngine:
    """Implements Propensity Score Matching (PSM), Difference-in-Differences (DiD), and Synthetic Control (SCM)."""

    # -------------------------------------------------------------------------
    # 1. Propensity Score Matching (PSM)
    # -------------------------------------------------------------------------
    @classmethod
    def propensity_score_matching(
        cls,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: List[str],
        caliper: Optional[float] = 0.05
    ) -> Dict[str, Any]:
        """Estimate Average Treatment Effect on the Treated (ATT) via 1:1 Nearest-Neighbor PSM."""
        clean_df = df[[treatment_col, outcome_col] + covariate_cols].dropna().copy()
        
        # 1. Fit Propensity Score Model
        X = clean_df[covariate_cols]
        T = clean_df[treatment_col].astype(int)
        Y = clean_df[outcome_col].astype(float)

        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X, T)
        clean_df["propensity_score"] = ps_model.predict_proba(X)[:, 1]

        treated_mask = clean_df[treatment_col] == 1
        control_mask = clean_df[treatment_col] == 0

        treated = clean_df[treated_mask]
        control = clean_df[control_mask]

        if len(treated) == 0 or len(control) == 0:
            raise ValueError("Both treatment and control groups must be populated.")

        # 2. Nearest-Neighbor Matching on Propensity Score
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(control[["propensity_score"]])
        distances, indices = nn.kneighbors(treated[["propensity_score"]])

        matched_treated_idx = []
        matched_control_idx = []

        for t_idx, (dist, c_idx) in enumerate(zip(distances, indices)):
            if caliper is None or dist[0] <= caliper:
                matched_treated_idx.append(treated.index[t_idx])
                matched_control_idx.append(control.index[c_idx[0]])

        if len(matched_treated_idx) == 0:
            raise RuntimeError("No pairs matched within the specified caliper.")

        matched_treated_outcomes = clean_df.loc[matched_treated_idx, outcome_col].values
        matched_control_outcomes = clean_df.loc[matched_control_idx, outcome_col].values

        # Pairwise treatment effect
        effects = matched_treated_outcomes - matched_control_outcomes
        att = float(np.mean(effects))
        se_att = float(stats.sem(effects)) if len(effects) > 1 else 0.0
        ci_lower = att - 1.96 * se_att
        ci_upper = att + 1.96 * se_att
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(att) / se_att))) if se_att > 0 else 0.0

        # Covariate balance (Standardized Mean Difference)
        balance = {}
        for col in covariate_cols:
            mean_t_pre = float(treated[col].mean())
            mean_c_pre = float(control[col].mean())
            var_t_pre = float(treated[col].var())
            pooled_sd = math.sqrt(max(0.0001, (var_t_pre + control[col].var()) / 2.0))
            smd_pre = abs(mean_t_pre - mean_c_pre) / pooled_sd

            # Post-match balance
            t_post = clean_df.loc[matched_treated_idx, col]
            c_post = clean_df.loc[matched_control_idx, col]
            smd_post = abs(t_post.mean() - c_post.mean()) / pooled_sd

            balance[col] = {
                "smd_before": round(smd_pre, 4),
                "smd_after": round(smd_post, 4),
                "balanced": bool(smd_post < 0.10)  # SMD < 0.10 is standard good balance
            }

        # DoWhy refutation if available
        dowhy_refutation = None
        if HAS_DOWHY:
            try:
                model = CausalModel(
                    data=clean_df,
                    treatment=treatment_col,
                    outcome=outcome_col,
                    common_causes=covariate_cols
                )
                identified_estimand = model.identify_effect()
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.propensity_score_matching"
                )
                refute = model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="random_common_cause"
                )
                dowhy_refutation = {
                    "method": "random_common_cause",
                    "new_effect": round(float(refute.new_effect), 4),
                    "p_value": round(float(refute.refutation_result.get("p_value", 0.5)), 4),
                    "passed": bool(abs(refute.new_effect - estimate.value) / max(0.001, abs(estimate.value)) < 0.20)
                }
            except Exception as e:
                logger.debug(f"DoWhy refutation skipped: {e}")

        return {
            "method": "propensity_score_matching",
            "att": round(att, 4),
            "standard_error": round(se_att, 4),
            "p_value": round(p_val, 6),
            "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "matched_pairs": len(matched_treated_idx),
            "total_treated": len(treated),
            "total_control": len(control),
            "covariate_balance": balance,
            "dowhy_refutation": dowhy_refutation
        }

    # -------------------------------------------------------------------------
    # 2. Difference-in-Differences (DiD)
    # -------------------------------------------------------------------------
    @classmethod
    def difference_in_differences(
        cls,
        df: pd.DataFrame,
        outcome_col: str,
        group_col: str,      # 1 = Treated, 0 = Control
        time_col: str,       # 1 = Post-treatment, 0 = Pre-treatment
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Estimate Average Treatment Effect via Difference-in-Differences OLS regression.
        Model: Y = b0 + b1*Treated + b2*Post + delta*(Treated * Post) + eps
        """
        import statsmodels.api as sm

        sub = df[[outcome_col, group_col, time_col] + (covariates or [])].dropna().copy()
        sub["interaction"] = sub[group_col] * sub[time_col]

        X_cols = [group_col, time_col, "interaction"] + (covariates or [])
        X = sm.add_constant(sub[X_cols])
        y = sub[outcome_col]

        model = sm.OLS(y, X).fit(cov_type="HC1")  # Robust Huber-White standard errors

        did_effect = float(model.params["interaction"])
        se = float(model.bse["interaction"])
        p_val = float(model.pvalues["interaction"])
        ci = model.conf_int().loc["interaction"]

        # Group means for visualization
        g_means = sub.groupby([group_col, time_col])[outcome_col].mean().to_dict()
        pre_c = g_means.get((0, 0), 0.0)
        post_c = g_means.get((0, 1), 0.0)
        pre_t = g_means.get((1, 0), 0.0)
        post_t = g_means.get((1, 1), 0.0)
        counterfactual_post_t = pre_t + (post_c - pre_c)

        return {
            "method": "difference_in_differences",
            "did_treatment_effect": round(did_effect, 4),
            "standard_error": round(se, 4),
            "t_statistic": round(float(model.tvalues["interaction"]), 4),
            "p_value": round(p_val, 6),
            "is_significant": bool(p_val < 0.05),
            "ci_95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "group_means": {
                "control_pre": round(float(pre_c), 4),
                "control_post": round(float(post_c), 4),
                "treated_pre": round(float(pre_t), 4),
                "treated_post": round(float(post_t), 4),
                "counterfactual_treated_post": round(float(counterfactual_post_t), 4)
            },
            "parallel_trends_lift": round(float(post_t - counterfactual_post_t), 4),
            "r_squared": round(float(model.rsquared), 4)
        }

    # -------------------------------------------------------------------------
    # 3. Synthetic Control Method (SCM)
    # -------------------------------------------------------------------------
    @classmethod
    def synthetic_control(
        cls,
        panel_df: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treated_unit_id: Any,
        intervention_time: Any
    ) -> Dict[str, Any]:
        """Construct a synthetic control unit via constrained convex optimization."""
        # Pivot panel: rows = time, columns = unit_id
        pivot = panel_df.pivot(index=time_col, columns=unit_col, values=outcome_col).dropna()

        all_times = sorted(pivot.index.tolist())
        pre_times = [t for t in all_times if t < intervention_time]
        post_times = [t for t in all_times if t >= intervention_time]

        if not pre_times or not post_times:
            raise ValueError("Panel must contain both pre- and post-intervention periods.")

        y_treated_pre = pivot.loc[pre_times, treated_unit_id].values
        donor_units = [u for u in pivot.columns if u != treated_unit_id]
        X_donors_pre = pivot.loc[pre_times, donor_units].values  # shape: (T_pre, J)

        num_donors = len(donor_units)

        # Objective: minimize || Y_treated - X_donors * W ||^2
        def objective(w):
            return np.sum((y_treated_pre - X_donors_pre @ w) ** 2)

        # Constraints: sum(w) = 1, w_j >= 0
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(num_donors)]
        w_init = np.ones(num_donors) / num_donors

        res = minimize(objective, w_init, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = res.x

        # Compute synthetic trajectory for all time
        synthetic_series = pivot[donor_units].values @ weights
        actual_series = pivot[treated_unit_id].values
        gaps = actual_series - synthetic_series

        post_idx = [i for i, t in enumerate(all_times) if t in post_times]
        att_post = float(np.mean(gaps[post_idx]))
        pre_rmse = math.sqrt(float(np.mean(gaps[[i for i, t in enumerate(all_times) if t in pre_times]] ** 2)))

        # Significant donor weights
        donor_weight_map = {
            str(unit): round(float(w), 4)
            for unit, w in zip(donor_units, weights)
            if w > 0.01
        }

        timeline = []
        for t, act, syn, gap in zip(all_times, actual_series, synthetic_series, gaps):
            timeline.append({
                "time": str(t),
                "actual": round(float(act), 3),
                "synthetic": round(float(syn), 3),
                "gap_effect": round(float(gap), 3),
                "is_post_intervention": bool(t >= intervention_time)
            })

        return {
            "method": "synthetic_control",
            "treated_unit": str(treated_unit_id),
            "intervention_time": str(intervention_time),
            "pre_intervention_rmse": round(pre_rmse, 4),
            "average_post_treatment_effect": round(att_post, 4),
            "donor_weights": donor_weight_map,
            "timeline": timeline
        }
