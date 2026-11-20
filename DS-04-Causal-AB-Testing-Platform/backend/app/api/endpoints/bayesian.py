"""Bayesian A/B Testing API Endpoints (Milestone 5)."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.bayesian_ab import BayesianABEngine
from backend.app.core.plain_english import PlainEnglishExplainer

router = APIRouter(prefix="/bayesian", tags=["Bayesian A/B Testing"])


class BayesianBetaBinomialRequest(BaseModel):
    conversions_a: int = Field(..., ge=0)
    visitors_a: int = Field(..., gt=0)
    conversions_b: int = Field(..., ge=0)
    visitors_b: int = Field(..., gt=0)
    prior_alpha: float = Field(default=1.0, gt=0.0)
    prior_beta: float = Field(default=1.0, gt=0.0)
    num_draws: int = Field(default=50000, gt=1000)
    rope_threshold: float = Field(default=0.001, ge=0.0)


class BayesianNormalRequest(BaseModel):
    data_a: List[float] = Field(..., min_length=2)
    data_b: List[float] = Field(..., min_length=2)
    num_draws: int = Field(default=30000, gt=1000)


@router.post("/beta-binomial")
def analyze_beta_binomial(req: BayesianBetaBinomialRequest):
    try:
        res = BayesianABEngine.analyze_beta_binomial(
            conversions_a=req.conversions_a,
            visitors_a=req.visitors_a,
            conversions_b=req.conversions_b,
            visitors_b=req.visitors_b,
            prior_alpha=req.prior_alpha,
            prior_beta=req.prior_beta,
            num_draws=req.num_draws,
            rope_threshold=req.rope_threshold
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["model"], res)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/continuous-normal")
def analyze_continuous_normal(req: BayesianNormalRequest):
    try:
        return BayesianABEngine.analyze_continuous_normal(
            data_a=req.data_a,
            data_b=req.data_b,
            num_draws=req.num_draws
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
