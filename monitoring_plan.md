# Post-Deployment Monitoring Plan

## 1. Data Drift Tracking
- Monitor PSI (Population Stability Index) for top 5 features weekly: `recency_days`, `monetary_180d`, `sessions_30d`, `frequency_180d`, and `days_since_signup`
- Compare incoming scoring batches against the 2025-09-30 training snapshot baseline distribution
- Alert if PSI > 0.2 for any feature (moderate drift) or PSI > 0.25 for two consecutive weeks (severe drift)
- Segment PSI by `loyalty_tier` and `city_tier` to catch localized shifts in Tier 1 Platinum customers

## 2. Prediction Distribution
- Track daily % of high/medium/low risk scores (thresholds: >0.7, 0.4–0.7, <0.4)
- Baseline from validation set: ~35% high, ~30% medium, ~35% low at snapshot time
- Alert if any bucket shifts >15% from baseline over a rolling 7-day window
- Plot weekly score histogram vs training `predicted_proba` distribution to detect score calibration drift

## 3. Business Outcomes
- Log CRM campaign response rate (open, click, repurchase) vs predicted risk buckets
- Track actual 60-day churn rate in a rolling window against model predictions by decile
- Compare retention offer ROI: cost per contacted customer vs incremental repurchase within 60 days
- Flag segments where high `monetary_180d` customers churn despite low predicted risk (false negative watchlist)

## 4. API Reliability & Errors
- Monitor 4xx/5xx error rates, request volume, and latency p95 on `/predict` and `/batch_predict`
- Alert on >5% error rate over 15 minutes or latency p95 >500ms sustained for 10 minutes
- Log validation failures (422) separately to detect upstream feature pipeline bugs
- Track `model_loaded=false` on `/health` as a critical page (model artifact missing or corrupt)

## 5. Retraining Triggers
- Retrain if: ROC-AUC drops >0.05 on recent labeled holdout data, PSI alert persists 2 weeks, or major product/pricing change occurs
- Schedule quarterly refresh regardless of drift (new snapshot feature pipeline + re-label 60-day churn)
- After retrain, run shadow scoring for 1 week before promoting new `model.pkl`
- Document threshold re-tuning (FN cost ₹2000 vs FP cost ₹200) in each model release

# Responsible Use Guidelines

## ✅ DO:
- Use scores to prioritize retention outreach, not auto-discount
- Combine with human review for high-value accounts (high `monetary_180d`, Platinum tier)
- Log CRM outcomes to improve model feedback loop and measure campaign effectiveness
- Respect `marketing_consent` and regional privacy requirements before contacting customers
- Cap outreach frequency to avoid fatigue among false-positive low-risk loyalists

## ❌ DO NOT:
- Use for credit scoring, pricing, or automated blocking of checkout or account access
- Assume causation (model predicts risk, not reasons)—explanations are heuristic summaries
- Deploy without privacy compliance & customer consent checks
- Target or exclude customers solely on proxy attributes (`age_group`, `city_tier`) without business justification
- Rely on the API during major assortment or pricing overhauls without revalidation
