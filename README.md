# Part 4: D2C Churn Scoring API

FastAPI service that serves the LightGBM churn model trained in **Part 3**, using **pre-snapshot features only** (snapshot date **2025-09-30**). Predictions estimate whether a customer will churn in the next 60 days (no purchase from 2025-10-01 to 2025-11-29).

flowchart TD
    subgraph P1["Part 1–2: Data & Segmentation"]
        A1[Raw D2C Data<br/>orders, customers, web events,<br/>support tickets, churn labels]
        A2[Feature Engineering<br/>RFM 180d, tickets 90d,<br/>web activity 30d]
        A3[Snapshot Date Guardrail<br/>2025-09-30]
        A4[RFM Segmentation<br/>customer segments]
        A1 --> A2 --> A3 --> A4
    end

    subgraph P3["Part 3: Model Training"]
        B1[rfm_modeling_snapshot.csv<br/>pre-snapshot features only]
        B2[Train / Validation / Test Split]
        B3[Preprocessing Pipeline<br/>median impute + one-hot encode]
        B4[LightGBM Classifier<br/>target: churn_next_60d]
        B5[Threshold Tuning<br/>FN cost ₹2000 vs FP ₹200]
        B6[Export model.pkl<br/>preprocessor + model + threshold]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6
    end

    subgraph P4["Part 4: FastAPI Churn Scoring API"]
        C0[Start Server<br/>uvicorn app.main:app]
        C1[Lifespan Startup<br/>joblib.load model.pkl]
        C2{Model loaded?}
        C3[GET /health<br/>status ok + model_loaded]
        C4[Client sends JSON<br/>CustomerFeatures]
        C5[Pydantic Validation<br/>types, ranges, required fields]
        C6{Valid input?}
        C7[Return 422<br/>validation errors]
        C8[Build DataFrame<br/>align feature columns]
        C9[Preprocessor.transform]
        C10[model.predict_proba]
        C11[Compute Outputs]
        C12[churn_probability 0–1]
        C13[predicted_class 0/1<br/>vs optimal threshold]
        C14[risk_level<br/>low / medium / high]
        C15[risk_explanation<br/>feature-backed text]
        C16[Return PredictionResponse<br/>POST /predict or /batch_predict]
        C17[Log errors<br/>500 on runtime failure]

        C0 --> C1 --> C2
        C2 -->|Yes| C3
        C2 -->|No| C17
        C4 --> C5 --> C6
        C6 -->|No| C7
        C6 -->|Yes| C8 --> C9 --> C10 --> C11
        C11 --> C12 --> C13 --> C14 --> C15 --> C16
    end

    subgraph OPS["Post-Deployment Monitoring & Retraining"]
        D1[Track Data Drift<br/>PSI on top features weekly]
        D2[Track Score Distribution<br/>high / medium / low risk %]
        D3[Track Business Outcomes<br/>CRM response vs predicted risk]
        D4[Track API Reliability<br/>4xx/5xx, latency p95]
        D5{Retrain trigger?<br/>AUC drop, persistent PSI,<br/>product change, quarterly}
        D6[Retrain in Part 3<br/>export new model.pkl]
        D7[Responsible Use<br/>prioritize outreach,<br/>human review for high LTV]
        D1 --> D5
        D2 --> D5
        D3 --> D5
        D4 --> D5
        D5 -->|Yes| D6 --> B6
        D5 -->|No| D7
    end

    subgraph QA["Validation & Testing"]
        E1[pytest tests/ -v]
        E2[test_health]
        E3[test_single_predict]
        E4[test_batch_predict]
        E5[test_invalid_input → 422]
        E1 --> E2 & E3 & E4 & E5
    end

    A3 --> B1
    B6 --> C1
    C16 --> D3
    C16 --> D4
    C3 --> E2
    C16 --> E3 & E4
    C7 --> E5

    style P3 fill:#e8f4fd,stroke:#1e88e5
    style P4 fill:#e8f5e9,stroke:#43a047
    style OPS fill:#fff3e0,stroke:#fb8c00
    style QA fill:#f3e5f5,stroke:#8e24aa


## Project Structure

```
part4_churn_api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, endpoints, risk explanations
│   └── schemas.py       # Pydantic input/output validation
├── tests/
│   ├── conftest.py
│   └── test_api.py
├── model.pkl            # Part 3 artifact (preprocessor + LightGBM + threshold)
├── requirements.txt
├── monitoring_plan.md
└── README.md
```

## Setup

```bash
cd part4_churn_api
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

> Use **`localhost`** in the browser, not `0.0.0.0`. The `--host 0.0.0.0` flag only tells the server to listen on all network interfaces.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health and model load status |
| `POST` | `/predict` | Score a single customer |
| `POST` | `/batch_predict` | Score multiple customers |

### GET `/health`

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST `/predict`

**Sample request:**
```json
{
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
  "last_visit_days_ago": 3
}
```

**Sample response:**
```json
{
  "churn_probability": 0.1245,
  "predicted_class": 0,
  "risk_level": "low",
  "risk_explanation": "Active buyer (3 orders) with 8 sessions and ₹1200 spend—strong retention."
}
```

### POST `/batch_predict`

**Sample request:**
```json
{
  "customers": [
    {
      "city_tier": "Tier 1",
      "age_group": "25-34",
      "acquisition_channel": "Instagram",
      "loyalty_tier": "Silver",
      "preferred_category": "Skincare",
      "marketing_consent": "Yes",
      "recency_days": 45,
      "frequency_180d": 3,
      "monetary_180d": 1200.0,
      "days_since_signup": 365,
      "sessions_30d": 8,
      "campaign_clicks_30d": 2
    },
    {
      "city_tier": "Tier 3",
      "age_group": "45-54",
      "acquisition_channel": "Organic",
      "loyalty_tier": "Bronze",
      "preferred_category": "Hair Care",
      "marketing_consent": "No",
      "recency_days": 130,
      "frequency_180d": 1,
      "monetary_180d": 280.0,
      "days_since_signup": 900,
      "sessions_30d": 0,
      "campaign_clicks_30d": 0
    }
  ]
}
```

**Sample response:**
```json
{
  "predictions": [
    {
      "churn_probability": 0.1245,
      "predicted_class": 0,
      "risk_level": "low",
      "risk_explanation": "Recent purchase (45d ago) and 8 sessions indicate strong retention likelihood."
    },
    {
      "churn_probability": 0.8123,
      "predicted_class": 1,
      "risk_level": "high",
      "risk_explanation": "Recency 130 days and zero sessions indicate strong disengagement and high churn risk."
    }
  ]
}
```

## Run Tests

```bash
pytest tests/ -v
```

## Model Source

- Trained in **Part 3** (`part3_churn_model/train_pipeline.py`) on `rfm_modeling_snapshot.csv`
- Saved as `model.pkl` containing: sklearn preprocessor, LightGBM classifier, optimal threshold (~0.20), and feature names
- Uses only features available **on or before 2025-09-30** — no post-snapshot leakage
- Top drivers: `recency_days`, `monetary_180d`, `days_since_signup`

## Monitoring & Responsible Use

See [`monitoring_plan.md`](monitoring_plan.md) for drift tracking, retraining triggers, and DO/DON'T guidelines.

## Quick Validation (curl)

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"city_tier\":\"Tier 1\",\"age_group\":\"25-34\",\"acquisition_channel\":\"Instagram\",\"loyalty_tier\":\"Silver\",\"preferred_category\":\"Skincare\",\"marketing_consent\":\"Yes\",\"recency_days\":45,\"frequency_180d\":3,\"monetary_180d\":1200.0,\"days_since_signup\":365,\"sessions_30d\":8,\"campaign_clicks_30d\":2}"
```
