"""FastAPI churn scoring service for D2C customer retention."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import schemas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model.pkl"

CAT_COLS = [
    "city_tier",
    "age_group",
    "acquisition_channel",
    "loyalty_tier",
    "preferred_category",
    "marketing_consent",
]
NUM_COLS = [
    "recency_days",
    "frequency_180d",
    "monetary_180d",
    "return_rate_180d",
    "avg_discount_pct_180d",
    "avg_rating_180d",
    "category_diversity_180d",
    "ticket_count_90d",
    "negative_ticket_rate_90d",
    "avg_resolution_hours_90d",
    "days_since_signup",
    "sessions_30d",
    "product_views_30d",
    "cart_adds_30d",
    "wishlist_adds_30d",
    "abandoned_carts_30d",
    "email_opens_30d",
    "campaign_clicks_30d",
    "last_visit_days_ago",
]
FEATURE_COLUMNS = CAT_COLS + NUM_COLS

model_artifact: dict[str, Any] | None = None


def generate_risk_explanation(features: dict[str, Any], prob: float) -> str:
    """Build a concise, feature-backed risk explanation (under 25 words)."""
    recency = int(features.get("recency_days", 0))
    sessions = int(features.get("sessions_30d", 0))
    tickets = int(features.get("ticket_count_90d", 0))
    frequency = int(features.get("frequency_180d", 0))
    monetary = float(features.get("monetary_180d", 0.0))
    neg_rate = float(features.get("negative_ticket_rate_90d", 0.0))
    clicks = int(features.get("campaign_clicks_30d", 0))

    if prob > 0.7:
        if recency > 90 and sessions == 0:
            return (
                f"Recency {recency} days and zero sessions indicate strong disengagement "
                f"and high churn risk."
            )
        if tickets > 0 and neg_rate > 0:
            return (
                f"{tickets} tickets with {neg_rate:.0%} negative rate and "
                f"{recency}d recency signal high attrition risk."
            )
        return (
            f"Recency {recency} days, {frequency} orders, and {sessions} sessions "
            f"point to elevated churn risk."
        )

    if prob >= 0.4:
        if tickets > 0:
            return (
                f"{tickets} recent tickets and {frequency} orders in 180d "
                f"suggest moderate attrition risk."
            )
        if recency > 60:
            return (
                f"Recency {recency} days with only {sessions} sessions "
                f"shows mild disengagement risk."
            )
        return (
            f"Mixed signals: ₹{monetary:.0f} spend, {frequency} orders, "
            f"{clicks} campaign clicks—moderate risk."
        )

    if frequency >= 2 and sessions >= 5:
        return (
            f"Active buyer ({frequency} orders) with {sessions} sessions "
            f"and ₹{monetary:.0f} spend—strong retention."
        )
    if recency <= 30:
        return (
            f"Recent purchase ({recency}d ago) and {sessions} sessions "
            f"indicate strong retention likelihood."
        )
    return (
        f"Low risk: {frequency} orders, {sessions} sessions, "
        f"recency {recency} days."
    )


def risk_level_from_probability(prob: float) -> str:
    """Map probability to business risk bucket."""
    if prob > 0.7:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"


def features_to_dataframe(customers: list[schemas.CustomerFeatures]) -> pd.DataFrame:
    """Convert validated Pydantic records to a training-aligned feature DataFrame."""
    rows = [customer.model_dump() for customer in customers]
    df = pd.DataFrame(rows)
    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)
    return df[FEATURE_COLUMNS]


def score_customers(customers: list[schemas.CustomerFeatures]) -> list[schemas.PredictionResponse]:
    """Run preprocessing and model inference for one or more customers."""
    if model_artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    preprocessor = model_artifact["preprocessor"]
    model = model_artifact["model"]
    threshold = float(model_artifact["threshold"])
    feature_names = model_artifact["feature_names"]

    raw_df = features_to_dataframe(customers)
    transformed = preprocessor.transform(raw_df)
    feature_df = pd.DataFrame(transformed, columns=feature_names)

    probabilities = model.predict_proba(feature_df)[:, 1]
    predictions: list[schemas.PredictionResponse] = []

    for customer, prob in zip(customers, probabilities):
        prob = float(prob)
        feature_dict = customer.model_dump()
        predicted_class = 1 if prob >= threshold else 0
        predictions.append(
            schemas.PredictionResponse(
                churn_probability=round(prob, 4),
                predicted_class=predicted_class,
                risk_level=risk_level_from_probability(prob),
                risk_explanation=generate_risk_explanation(feature_dict, prob),
            )
        )

    return predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the serialized Part 3 model artifact on startup."""
    global model_artifact
    try:
        model_artifact = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as exc:
        model_artifact = None
        logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)
    yield
    model_artifact = None


app = FastAPI(
    title="D2C Churn Scoring API",
    description="Scores customer churn risk using pre-snapshot features (2025-09-30).",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error during prediction."},
    )


@app.get("/")
def root() -> dict[str, Any]:
    """API landing page with available endpoints."""
    return {
        "service": "D2C Churn Scoring API",
        "status": "running",
        "model_loaded": model_artifact is not None,
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "batch_predict": "POST /batch_predict",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint for load balancers and deployment probes."""
    return {"status": "ok", "model_loaded": model_artifact is not None}


@app.post("/predict", response_model=schemas.PredictionResponse)
def predict(customer: schemas.CustomerFeatures) -> schemas.PredictionResponse:
    """Score a single customer for 60-day churn risk."""
    try:
        results = score_customers([customer])
        return results[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@app.post("/batch_predict", response_model=schemas.BatchPredictionResponse)
def batch_predict(
    request: schemas.BatchPredictionRequest,
) -> schemas.BatchPredictionResponse:
    """Score multiple customers in one request."""
    try:
        predictions = score_customers(request.customers)
        return schemas.BatchPredictionResponse(predictions=predictions)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Batch prediction failed.") from exc
