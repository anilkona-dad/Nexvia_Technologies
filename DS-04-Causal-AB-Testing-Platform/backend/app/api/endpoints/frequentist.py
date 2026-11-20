"""Frequentist Hypothesis Testing API Endpoints (Milestone 3)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.frequentist_tests import FrequentistEngine
from backend.app.core.plain_english import PlainEnglishExplainer
from backend.app.data_loader import BenchmarkDataLoader

router = APIRouter(prefix="/frequentist", tags=["Frequentist Testing"])


class TTestRequest(BaseModel):
    control_data: List[float] = Field(..., min_length=2)
    variant_data: List[float] = Field(..., min_length=2)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    equal_var: bool = Field(default=False)


class ChiSquareRequest(BaseModel):
    conversions_control: int = Field(..., ge=0)
    visitors_control: int = Field(..., gt=0)
    conversions_variant: int = Field(..., ge=0)
    visitors_variant: int = Field(..., gt=0)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)


class MultipleCorrectionRequest(BaseModel):
    test_results: List[Dict[str, Any]]
    method: str = Field(default="benjamini_hochberg", description="benjamini_hochberg | bonferroni")
    q_threshold: float = Field(default=0.05, gt=0.0, lt=0.5)


@router.post("/t-test")
def run_t_test(req: TTestRequest):
    try:
        res = FrequentistEngine.two_sample_t_test(
            control_data=req.control_data,
            variant_data=req.variant_data,
            alpha=req.alpha,
            equal_var=req.equal_var
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["test_type"], res)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chi-square")
def run_chi_square(req: ChiSquareRequest):
    try:
        res = FrequentistEngine.chi_square_test(
            conversions_control=req.conversions_control,
            visitors_control=req.visitors_control,
            conversions_variant=req.conversions_variant,
            visitors_variant=req.visitors_variant,
            alpha=req.alpha
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["test_type"], res)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/multiple-testing-correction")
def apply_correction(req: MultipleCorrectionRequest):
    try:
        return FrequentistEngine.apply_multiple_testing_correction(
            test_results=req.test_results,
            method=req.method,
            q_threshold=req.q_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/benchmark-cookie-cats")
def analyze_cookie_cats_benchmark():
    """Analyze the Cookie Cats mobile game retention A/B test benchmark."""
    try:
        df = BenchmarkDataLoader.load_cookie_cats_ab_data()
        c = df[df["version"] == "gate_30"]
        v = df[df["version"] == "gate_40"]

        # 1. 7-day retention chi-square
        chi_res = FrequentistEngine.chi_square_test(
            conversions_control=int(c["retention_7"].sum()),
            visitors_control=len(c),
            conversions_variant=int(v["retention_7"].sum()),
            visitors_variant=len(v)
        )
        chi_res["plain_english"] = PlainEnglishExplainer.explain(chi_res["test_type"], chi_res)

        # 2. Game rounds continuous Welch t-test
        t_res = FrequentistEngine.two_sample_t_test(
            control_data=c["sum_gamerounds"].tolist(),
            variant_data=v["sum_gamerounds"].tolist()
        )
        t_res["plain_english"] = PlainEnglishExplainer.explain(t_res["test_type"], t_res)

        return {
            "dataset": "Cookie Cats Mobile Game (Gate 30 vs Gate 40)",
            "sample_size": len(df),
            "retention_7_day_test": chi_res,
            "gamerounds_played_test": t_res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
