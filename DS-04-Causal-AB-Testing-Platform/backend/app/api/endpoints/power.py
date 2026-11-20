"""Power & Sample Size Calculator API Endpoints (Milestone 2)."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.core.power_calc import PowerCalculator

router = APIRouter(prefix="/power", tags=["Power & Sample Size"])


class SampleSizeProportionRequest(BaseModel):
    baseline_rate: float = Field(..., gt=0.0, lt=1.0, description="Baseline conversion rate, e.g. 0.10")
    mde_relative: float = Field(..., description="Minimum Detectable Effect relative to baseline, e.g. 0.05 for +5%")
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    power: float = Field(default=0.80, gt=0.5, lt=1.0)
    ratio: float = Field(default=1.0, gt=0.0)


class SampleSizeContinuousRequest(BaseModel):
    mean: float = Field(..., description="Baseline mean value, e.g. average order value")
    std_dev: float = Field(..., gt=0.0, description="Standard deviation of metric")
    mde_relative: float = Field(..., description="Relative minimum detectable effect")
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    power: float = Field(default=0.80, gt=0.5, lt=1.0)
    ratio: float = Field(default=1.0, gt=0.0)


class PowerCurveRequest(BaseModel):
    baseline_rate: float = Field(default=0.10, gt=0.0, lt=1.0)
    mde_relative: float = Field(default=0.05)
    alpha: float = Field(default=0.05)
    sample_sizes: Optional[List[int]] = None


class MDERequest(BaseModel):
    baseline_rate: float = Field(default=0.10, gt=0.0, lt=1.0)
    sample_size_per_variant: int = Field(..., gt=10)
    alpha: float = Field(default=0.05)
    power: float = Field(default=0.80)


@router.post("/sample-size-proportion")
def calculate_sample_size_proportion(req: SampleSizeProportionRequest):
    try:
        return PowerCalculator.calculate_sample_size_proportion(
            baseline_rate=req.baseline_rate,
            mde_relative=req.mde_relative,
            alpha=req.alpha,
            power=req.power,
            ratio=req.ratio
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sample-size-continuous")
def calculate_sample_size_continuous(req: SampleSizeContinuousRequest):
    try:
        return PowerCalculator.calculate_sample_size_continuous(
            mean=req.mean,
            std_dev=req.std_dev,
            mde_relative=req.mde_relative,
            alpha=req.alpha,
            power=req.power,
            ratio=req.ratio
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/power-curve")
def generate_power_curve(req: PowerCurveRequest):
    try:
        return PowerCalculator.generate_power_curve(
            baseline_rate=req.baseline_rate,
            sample_sizes=req.sample_sizes,
            alpha=req.alpha,
            mde_relative=req.mde_relative
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/mde")
def calculate_mde(req: MDERequest):
    try:
        return PowerCalculator.calculate_mde(
            baseline_rate=req.baseline_rate,
            sample_size_per_variant=req.sample_size_per_variant,
            alpha=req.alpha,
            power=req.power
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
