"""Pytest fixtures for API integration tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """Shared TestClient for all API tests."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer_payload() -> dict:
    """Sample customer payload aligned with rfm_modeling_snapshot.csv schema."""
    return {
        "city_tier": "Tier 1",
        "age_group": "25-34",
        "acquisition_channel": "Instagram",
        "loyalty_tier": "Silver",
        "preferred_category": "Skincare",
        "marketing_consent": "Yes",
        "recency_days": 45,
        "frequency_180d": 3,
        "monetary_180d": 1200.0,
        "return_rate_180d": 0.0,
        "avg_discount_pct_180d": 0.15,
        "avg_rating_180d": 4.2,
        "category_diversity_180d": 2,
        "ticket_count_90d": 0,
        "negative_ticket_rate_90d": 0.0,
        "avg_resolution_hours_90d": 0.0,
        "days_since_signup": 365,
        "sessions_30d": 8,
        "product_views_30d": 25,
        "cart_adds_30d": 4,
        "wishlist_adds_30d": 2,
        "abandoned_carts_30d": 1,
        "email_opens_30d": 5,
        "campaign_clicks_30d": 2,
        "last_visit_days_ago": 3,
    }
