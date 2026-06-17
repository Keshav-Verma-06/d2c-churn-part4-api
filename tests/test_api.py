"""Integration tests for the D2C Churn Scoring API."""


def test_health(client):
    """GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_single_predict(client, valid_customer_payload):
    """POST /predict returns all required prediction fields."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "churn_probability",
        "predicted_class",
        "risk_level",
        "risk_explanation",
    }
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_class"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high")
    assert isinstance(body["risk_explanation"], str)
    assert len(body["risk_explanation"]) > 0


def test_batch_predict(client, valid_customer_payload):
    """POST /batch_predict scores multiple customers."""
    high_risk = {
        **valid_customer_payload,
        "recency_days": 120,
        "sessions_30d": 0,
        "frequency_180d": 1,
        "monetary_180d": 250.0,
    }
    payload = {"customers": [valid_customer_payload, high_risk]}
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    for prediction in body["predictions"]:
        assert 0.0 <= prediction["churn_probability"] <= 1.0
        assert prediction["risk_level"] in ("low", "medium", "high")


def test_invalid_input(client, valid_customer_payload):
    """POST /predict with invalid values returns 422."""
    invalid = {**valid_customer_payload, "recency_days": -5}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

    missing_required = {k: v for k, v in valid_customer_payload.items() if k != "recency_days"}
    response = client.post("/predict", json=missing_required)
    assert response.status_code == 422
