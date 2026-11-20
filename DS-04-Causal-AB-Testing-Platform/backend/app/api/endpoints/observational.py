"""Observational Causal Inference API Endpoints (Milestone 6)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

from backend.app.core.causal_observational import ObservationalCausalEngine
from backend.app.core.plain_english import PlainEnglishExplainer
from backend.app.data_loader import BenchmarkDataLoader

router = APIRouter(prefix="/causal", tags=["Observational Causal Inference"])


class PSMRequest(BaseModel):
    data: List[Dict[str, Any]]
    treatment_col: str
    outcome_col: str
    covariate_cols: List[str]
    caliper: Optional[float] = 0.05


class DiDRequest(BaseModel):
    data: List[Dict[str, Any]]
    outcome_col: str
    group_col: str
    time_col: str
    covariates: Optional[List[str]] = None


class SyntheticControlRequest(BaseModel):
    panel_data: List[Dict[str, Any]]
    unit_col: str
    time_col: str
    outcome_col: str
    treated_unit_id: Any
    intervention_time: Any


@router.post("/psm")
def run_psm(req: PSMRequest):
    try:
        df = pd.DataFrame(req.data)
        res = ObservationalCausalEngine.propensity_score_matching(
            df=df,
            treatment_col=req.treatment_col,
            outcome_col=req.outcome_col,
            covariate_cols=req.covariate_cols,
            caliper=req.caliper
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["method"], res)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/did")
def run_did(req: DiDRequest):
    try:
        df = pd.DataFrame(req.data)
        res = ObservationalCausalEngine.difference_in_differences(
            df=df,
            outcome_col=req.outcome_col,
            group_col=req.group_col,
            time_col=req.time_col,
            covariates=req.covariates
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["method"], res)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/synthetic-control")
def run_synthetic_control(req: SyntheticControlRequest):
    try:
        df = pd.DataFrame(req.panel_data)
        res = ObservationalCausalEngine.synthetic_control(
            panel_df=df,
            unit_col=req.unit_col,
            time_col=req.time_col,
            outcome_col=req.outcome_col,
            treated_unit_id=req.treated_unit_id,
            intervention_time=req.intervention_time
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["method"], res)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Benchmark Endpoints
@router.get("/benchmark-card-krueger")
def run_card_krueger_benchmark():
    """Card & Krueger (1994) Minimum Wage Experiment (Difference-in-Differences)."""
    try:
        df = BenchmarkDataLoader.load_card_krueger_did_data()
        res = ObservationalCausalEngine.difference_in_differences(
            df=df,
            outcome_col="fte_employment",
            group_col="treated_state_nj",
            time_col="post_period"
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["method"], res)
        res["dataset_description"] = (
            "Card & Krueger (1994): Employment in fast food restaurants across New Jersey (Treated) "
            "vs Eastern Pennsylvania (Control) before and after NJ minimum wage increase."
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark-lalonde")
def run_lalonde_benchmark():
    """LaLonde (1986) National Supported Work Demonstration (PSM)."""
    try:
        df = BenchmarkDataLoader.load_lalonde_psm_data()
        res = ObservationalCausalEngine.propensity_score_matching(
            df=df,
            treatment_col="treat",
            outcome_col="re78_outcome",
            covariate_cols=["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"],
            caliper=0.10
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["method"], res)
        res["dataset_description"] = (
            "LaLonde (1986): Observational evaluation of job training program effect on post-intervention earnings."
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark-prop99")
def run_prop99_benchmark():
    """California Proposition 99 Tobacco Control (Synthetic Control Method)."""
    try:
        df = BenchmarkDataLoader.load_california_prop99_scm_data()
        res = ObservationalCausalEngine.synthetic_control(
            panel_df=df,
            unit_col="state",
            time_col="year",
            outcome_col="cigsale",
            treated_unit_id="California",
            intervention_time=1988
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["method"], res)
        res["dataset_description"] = (
            "California Proposition 99: Synthetic control evaluation of large tobacco tax and anti-smoking campaign in 1988."
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
