## API Documentation

Once the server is running, you can access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

The Swagger UI allows you to:
- View all available endpoints
- Test API calls directly from your browser
- See request/response schemas
- Download the OpenAPI specification

# Part 4: D2C Churn Scoring API

FastAPI service that serves the LightGBM churn model trained in **Part 3**, using **pre-snapshot features only** (snapshot date **2025-09-30**). Predictions estimate whether a customer will churn in the next 60 days (no purchase from 2025-10-01 to 2025-11-29).

<img width="2148" height="8192" alt="image" src="https://github.com/user-attachments/assets/2df67ef6-af05-4c75-97b5-d187ac2e0c95" />

Part	Command to Run	Expected Result
**Part 1**	`jupyter execute eda_audit.ipynb`	Notebook runs top-to-bottom, charts regenerate
**Part 2**	`python -c "import pandas as pd; df=pd.read_csv('segments.csv'); print(df.shape)"`	Outputs `(2400, X)` with required columns
**Part 3**	`python -c "import joblib; m=joblib.load('model.pkl'); print(type(m))"`	Prints model type (e.g., `<class 'lightgbm.sklearn.LGBMClassifier'>`)
**Part 4**	`pytest tests/ -v` then `uvicorn app.main:app --port 8000` + curl test	4/4 tests pass; `/predict` returns dynamic JSON with all 4 fields
<img width="798" height="106" alt="image" src="https://github.com/user-attachments/assets/1ce34e12-bb93-42e5-93f7-a826aaef193b" />


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

## Public Repository

https://github.com/Keshav-Verma-06/d2c-churn-part4-api

## Quick Validation (curl)

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"city_tier\":\"Tier 1\",\"age_group\":\"25-34\",\"acquisition_channel\":\"Instagram\",\"loyalty_tier\":\"Silver\",\"preferred_category\":\"Skincare\",\"marketing_consent\":\"Yes\",\"recency_days\":45,\"frequency_180d\":3,\"monetary_180d\":1200.0,\"days_since_signup\":365,\"sessions_30d\":8,\"campaign_clicks_30d\":2}"
```
