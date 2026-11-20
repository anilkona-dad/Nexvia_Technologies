"""Sequential Testing & Peeking Protection API Endpoints (Milestone 4)."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.plain_english import PlainEnglishExplainer
from backend.app.core.sequential_tests import SequentialEngine

router = APIRouter(prefix="/sequential", tags=["Sequential Testing"])


class MSPRTRequest(BaseModel):
    conversions_a: List[int] = Field(..., description="Binary stream of observations for control")
    conversions_b: List[int] = Field(..., description="Binary stream of observations for variant")
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    tau: float = Field(default=0.05, gt=0.0, description="Mixing parameter for prior lift variance")


class AlphaSpendingRequest(BaseModel):
    total_planned_sample_size: int = Field(..., gt=100)
    interim_sample_sizes: List[int] = Field(..., min_length=1)
    alpha: float = Field(default=0.05)


class PeekingSimRequest(BaseModel):
    sample_size: int = Field(default=1500, gt=100)
    peeks: int = Field(default=15, gt=2)
    num_simulations: int = Field(default=300, gt=10)


@router.post("/msprt")
def run_msprt(req: MSPRTRequest):
    try:
        res = SequentialEngine.msprt_proportions(
            conversions_a=req.conversions_a,
            conversions_b=req.conversions_b,
            alpha=req.alpha,
            tau=req.tau
        )
        res["plain_english"] = PlainEnglishExplainer.explain(res["test_type"], res)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/alpha-spending")
def calculate_alpha_spending(req: AlphaSpendingRequest):
    try:
        return SequentialEngine.alpha_spending_obrien_fleming(
            total_planned_sample_size=req.total_planned_sample_size,
            interim_sample_sizes=req.interim_sample_sizes,
            alpha=req.alpha
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/simulate-peeking-problem")
def simulate_peeking_problem(req: PeekingSimRequest):
    """Simulate Type I error rate when teams peek at p-values repeatedly versus mSPRT."""
    try:
        return SequentialEngine.simulate_peeking_problem(
            sample_size=req.sample_size,
            peeks=req.peeks,
            num_simulations=req.num_simulations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
