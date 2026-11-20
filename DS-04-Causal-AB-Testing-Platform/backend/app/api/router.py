"""Unified API Router mounting all statistical sub-routers."""

from fastapi import APIRouter

from backend.app.api.endpoints.power import router as power_router
from backend.app.api.endpoints.frequentist import router as frequentist_router
from backend.app.api.endpoints.sequential import router as sequential_router
from backend.app.api.endpoints.bayesian import router as bayesian_router
from backend.app.api.endpoints.observational import router as observational_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(power_router)
api_router.include_router(frequentist_router)
api_router.include_router(sequential_router)
api_router.include_router(bayesian_router)
api_router.include_router(observational_router)
