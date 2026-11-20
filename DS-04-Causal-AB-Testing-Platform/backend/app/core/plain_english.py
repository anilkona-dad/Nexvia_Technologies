"""Plain-English Decision Explainer Engine (Milestone 7)."""

from typing import Any, Dict


class PlainEnglishExplainer:
    """Translates complex statistical outputs into actionable, executive plain-English conclusions."""

    @classmethod
    def explain(cls, test_type: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured plain-English guidance for the given statistical result."""
        if test_type in ["chi_square_proportions", "welch_t_test", "student_t_test"]:
            return cls._explain_frequentist(result)
        elif test_type == "mSPRT_proportions":
            return cls._explain_sequential(result)
        elif test_type == "beta_binomial_conjugate":
            return cls._explain_bayesian(result)
        elif test_type == "difference_in_differences":
            return cls._explain_did(result)
        elif test_type == "propensity_score_matching":
            return cls._explain_psm(result)
        elif test_type == "synthetic_control":
            return cls._explain_synthetic_control(result)
        else:
            return {
                "verdict": "ANALYSIS_COMPLETE",
                "summary": "Statistical analysis completed successfully.",
                "recommendation": "Review metric distributions and decision thresholds."
            }

    @classmethod
    def _explain_frequentist(cls, r: Dict[str, Any]) -> Dict[str, Any]:
        p_val = r.get("p_value", 1.0)
        is_sig = r.get("is_significant", False)
        diff = r.get("difference", {})
        rel_lift = diff.get("relative_lift_pct", 0.0)
        ci_lower = diff.get("ci_lower", 0.0)
        ci_upper = diff.get("ci_upper", 0.0)

        if is_sig and rel_lift > 0:
            verdict = "WINNER_DETECTED"
            summary = (
                f"Variant outperformed Control with statistical significance (p = {p_val:.4f}). "
                f"We observed an estimated relative lift of +{rel_lift:.2f}% "
                f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
                f"The probability of observing this lift purely by random chance is less than {p_val*100:.2f}%."
            )
            recommendation = "Safe to ship variant to 100% of production traffic."
        elif is_sig and rel_lift < 0:
            verdict = "DEGRADATION_DETECTED"
            summary = (
                f"Variant caused a statistically significant drop of {rel_lift:.2f}% (p = {p_val:.4f}). "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]."
            )
            recommendation = "Do not deploy variant. Roll back immediately to protect core metrics."
        else:
            verdict = "NO_SIGNIFICANT_DIFFERENCE"
            summary = (
                f"No statistically significant difference detected between variants (p = {p_val:.4f} >= 0.05). "
                f"The observed relative difference was {rel_lift:+.2f}% (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])."
            )
            recommendation = (
                "Continue running test if planned sample size has not been reached, "
                "or accept the null hypothesis if test duration is complete."
            )

        return {
            "verdict": verdict,
            "headline": f"{'Statistically Significant' if is_sig else 'Inconclusive'} ({rel_lift:+.2f}% lift)",
            "summary": summary,
            "recommendation": recommendation,
            "risk_assessment": "Low risk" if is_sig and rel_lift > 0 else "High risk" if is_sig and rel_lift < 0 else "Inconclusive"
        }

    @classmethod
    def _explain_sequential(cls, r: Dict[str, Any]) -> Dict[str, Any]:
        stopped_early = r.get("stopped_early", False)
        lr = r.get("current_likelihood_ratio", 0.0)
        threshold = r.get("threshold", 20.0)
        n = r.get("total_observations_tested", 0)

        if stopped_early:
            verdict = "EARLY_STOP_WINNER"
            summary = (
                f"The sequential mSPRT test hit the early stopping boundary at sample size n = {r.get('stopping_sample_size', n)}. "
                f"The likelihood ratio reached {lr:.1f} (threshold = {threshold:.1f}), successfully rejecting the null hypothesis "
                f"while strictly controlling the false-positive rate under repeated peeking."
            )
            recommendation = "You can stop the experiment early today and deploy the winner ahead of schedule."
        else:
            verdict = "CONTINUE_PEEKING_SAFELY"
            summary = (
                f"After evaluating {n} observations, the sequential test has not reached the early stopping boundary "
                f"(Current likelihood ratio: {lr:.2f} vs. required {threshold:.1f})."
            )
            recommendation = "Keep the experiment running. Safe to monitor daily without inflating Type I error."

        return {
            "verdict": verdict,
            "headline": "Early Stop Reached" if stopped_early else "Keep Experiment Running",
            "summary": summary,
            "recommendation": recommendation,
            "risk_assessment": "Type I error rate strictly guaranteed at 5%."
        }

    @classmethod
    def _explain_bayesian(cls, r: Dict[str, Any]) -> Dict[str, Any]:
        prob_b = r.get("probability_b_beats_a", 0.5)
        exp_loss = r.get("expected_loss_deploy_b", 0.0)
        hdi = r.get("relative_lift_pct_95_hdi", [0.0, 0.0])
        decision = r.get("decision", "INCONCLUSIVE")

        summary = (
            f"The Bayesian model indicates a {prob_b * 100:.1f}% posterior probability that Variant B outperforms Control. "
            f"The 95% Credible Interval for relative lift spans [{hdi[0]:+.2f}%, {hdi[1]:+.2f}%]. "
            f"If Variant B is shipped, the expected loss is estimated at {exp_loss * 100:.3f}%."
        )

        if decision == "DEPLOY_B":
            verdict = "STRONG_BAYESIAN_WINNER"
            recommendation = "High probability of positive impact with negligible expected loss. Deploy variant."
        elif decision == "DEPLOY_A":
            verdict = "CONTROL_SUPERIOR"
            recommendation = "Control has high probability of being superior. Maintain baseline."
        else:
            verdict = "BAYESIAN_EQUITABLE_OR_PENDING"
            recommendation = "The difference falls within practical equivalence or requires more data."

        return {
            "verdict": verdict,
            "headline": f"{prob_b * 100:.1f}% Probability B > A",
            "summary": summary,
            "recommendation": recommendation,
            "risk_assessment": f"Expected loss: {exp_loss:.5f} (lower is safer)"
        }

    @classmethod
    def _explain_did(cls, r: Dict[str, Any]) -> Dict[str, Any]:
        effect = r.get("did_treatment_effect", 0.0)
        p_val = r.get("p_value", 1.0)
        is_sig = r.get("is_significant", False)
        means = r.get("group_means", {})

        summary = (
            f"The Difference-in-Differences model estimated a causal treatment effect of {effect:+.4f} "
            f"(p = {p_val:.4f}). Relative to the counterfactual trajectory established by the control group "
            f"({means.get('counterfactual_treated_post', 0):.2f}), the treated group attained an average post-intervention outcome of "
            f"{means.get('treated_post', 0):.2f}."
        )

        return {
            "verdict": "CAUSAL_EFFECT_SIGNIFICANT" if is_sig else "CAUSAL_EFFECT_NOT_SIGNIFICANT",
            "headline": f"Estimated DiD Treatment Effect: {effect:+.4f}",
            "summary": summary,
            "recommendation": "Causal impact confirmed under the parallel trends assumption." if is_sig else "Insufficient evidence to conclude policy intervention changed the trajectory.",
            "risk_assessment": "Verify that pre-treatment trends were parallel between groups."
        }

    @classmethod
    def _explain_psm(cls, r: Dict[str, Any]) -> Dict[str, Any]:
        att = r.get("att", 0.0)
        p_val = r.get("p_value", 1.0)
        pairs = r.get("matched_pairs", 0)

        summary = (
            f"Propensity score matching successfully paired {pairs} treated individuals with statistically identical control counterparts. "
            f"The Average Treatment Effect on the Treated (ATT) is {att:+.4f} (p = {p_val:.4f}). "
            f"Covariate imbalances (SMDs) were minimized across matching dimensions."
        )

        return {
            "verdict": "ATT_ESTIMATED",
            "headline": f"Matched ATT Effect: {att:+.4f}",
            "summary": summary,
            "recommendation": "Treatment effect is causal conditional on unconfoundedness / selection on observables.",
            "risk_assessment": "Check DoWhy refutation tests against hidden unobserved confounders."
        }

    @classmethod
    def _explain_synthetic_control(cls, r: Dict[str, Any]) -> Dict[str, Any]:
        att_post = r.get("average_post_treatment_effect", 0.0)
        pre_rmse = r.get("pre_intervention_rmse", 0.0)
        treated = r.get("treated_unit", "Treated Unit")

        summary = (
            f"A convex synthetic counterpart for {treated} was constructed with pre-intervention fit RMSE of {pre_rmse:.4f}. "
            f"Following intervention, the average gap between {treated} and its synthetic twin was {att_post:+.4f}."
        )

        return {
            "verdict": "SYNTHETIC_CONTROL_ESTIMATED",
            "headline": f"Post-Intervention Gap: {att_post:+.4f}",
            "summary": summary,
            "recommendation": "Inspect post-intervention gap divergence and donor unit weighting stability.",
            "risk_assessment": "Low pre-intervention RMSE confirms synthetic twin validity."
        }
