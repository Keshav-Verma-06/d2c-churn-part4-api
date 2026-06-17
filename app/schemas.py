"""Pydantic schemas for churn prediction API requests and responses."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Pre-snapshot customer features aligned with Part 3 training columns (2025-09-30)."""

    # Categorical profile fields
    city_tier: str = Field(
        ...,
        description="Customer city tier (e.g., Tier 1, Tier 2, Tier 3).",
        examples=["Tier 1"],
    )
    age_group: str = Field(
        ...,
        description="Customer age band.",
        examples=["25-34"],
    )
    acquisition_channel: str = Field(
        ...,
        description="Channel through which the customer was acquired.",
        examples=["Instagram"],
    )
    loyalty_tier: str = Field(
        ...,
        description="Current loyalty program tier.",
        examples=["Silver"],
    )
    preferred_category: str = Field(
        ...,
        description="Primary product category preference.",
        examples=["Skincare"],
    )
    marketing_consent: str = Field(
        ...,
        description="Whether the customer opted in to marketing (Yes/No).",
        examples=["Yes"],
    )

    # RFM and order behavior (180-day window)
    recency_days: int = Field(
        ...,
        ge=0,
        description="Days since last order as of snapshot date.",
    )
    frequency_180d: int = Field(
        ...,
        ge=0,
        description="Number of orders in the last 180 days.",
    )
    monetary_180d: float = Field(
        ...,
        ge=0.0,
        description="Total spend in INR over the last 180 days.",
    )
    return_rate_180d: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of orders returned in the last 180 days (0–1).",
    )
    avg_discount_pct_180d: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average discount percentage applied on orders (0–1).",
    )
    avg_rating_180d: Optional[float] = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Average product rating given in the last 180 days.",
    )
    category_diversity_180d: int = Field(
        default=1,
        ge=0,
        description="Count of distinct product categories purchased in 180 days.",
    )

    # Support tickets (90-day window)
    ticket_count_90d: int = Field(
        default=0,
        ge=0,
        description="Support tickets opened in the last 90 days.",
    )
    negative_ticket_rate_90d: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of tickets with negative sentiment (0–1).",
    )
    avg_resolution_hours_90d: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        description="Average ticket resolution time in hours.",
    )

    # Customer tenure
    days_since_signup: int = Field(
        ...,
        ge=0,
        description="Days between signup and snapshot date.",
    )

    # Web / app engagement (30-day window)
    sessions_30d: int = Field(
        default=0,
        ge=0,
        description="Web or app sessions in the last 30 days.",
    )
    product_views_30d: Optional[int] = Field(
        default=0,
        ge=0,
        description="Product detail page views in the last 30 days.",
    )
    cart_adds_30d: Optional[int] = Field(
        default=0,
        ge=0,
        description="Add-to-cart events in the last 30 days.",
    )
    wishlist_adds_30d: Optional[int] = Field(
        default=0,
        ge=0,
        description="Wishlist additions in the last 30 days.",
    )
    abandoned_carts_30d: Optional[int] = Field(
        default=0,
        ge=0,
        description="Abandoned cart events in the last 30 days.",
    )
    email_opens_30d: Optional[int] = Field(
        default=0,
        ge=0,
        description="Marketing email opens in the last 30 days.",
    )
    campaign_clicks_30d: int = Field(
        default=0,
        ge=0,
        description="Campaign link clicks in the last 30 days.",
    )
    last_visit_days_ago: Optional[int] = Field(
        default=0,
        ge=0,
        description="Days since last web/app visit as of snapshot.",
    )


class PredictionResponse(BaseModel):
    """Single-customer churn prediction output."""

    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of churn in the next 60 days.",
    )
    predicted_class: Literal[0, 1] = Field(
        ...,
        description="Binary churn prediction (1 = at-risk) using the Part 3 optimal threshold.",
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Business risk bucket derived from churn probability.",
    )
    risk_explanation: str = Field(
        ...,
        description="Short, feature-backed explanation of the risk assessment.",
    )


class BatchPredictionRequest(BaseModel):
    """Batch scoring request containing multiple customers."""

    customers: List[CustomerFeatures] = Field(
        ...,
        min_length=1,
        description="List of customer feature records to score.",
    )


class BatchPredictionResponse(BaseModel):
    """Batch scoring response with one prediction per input customer."""

    predictions: List[PredictionResponse] = Field(
        ...,
        description="Churn predictions aligned with the input customer order.",
    )
