# Account Health Scoring Tool

A PM-facing dashboard for predicting B2B SaaS account churn risk, with
per-account SHAP explanations. Built for the dissertation "Predictive
Health Scoring for Product-Led Expansion."

## What it does

- Loads five CSVs (accounts, subscriptions, feature_usage,
  support_tickets, churn_events) in the RavenStack schema
- Trains an XGBoost classifier to predict churn
- Scores every account into a health quadrant (Healthy / At Risk /
  Critical / Lost) based on predicted churn risk and MRR
- Lets a user look up any account and see a live SHAP explanation of
  what's driving its risk score

## Running locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place your five CSVs in the same folder as `health_scoring_app.py`
   (or use the bundled demo data toggle in the sidebar)
3. Run:
   ```
   streamlit run health_scoring_app.py
   ```
4. Open the local URL Streamlit prints (usually
   `http://localhost:8501`)

## Deploying for free (so anyone can use it via a URL)

1. Create a new GitHub repository and push these three files:
   `health_scoring_app.py`, `requirements.txt`, this `README.md`
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub
3. Click "New app," select your repository and
   `health_scoring_app.py` as the main file
4. Deploy — Streamlit Community Cloud builds and hosts it for free,
   giving you a public URL

## Using your own data

Uncheck "Use bundled demo data" in the sidebar and upload five CSVs
matching this schema:

| File | Required columns |
|---|---|
| accounts.csv | account_id, industry, country, plan_tier, referral_source, ... |
| subscriptions.csv | subscription_id, account_id, mrr_amount, seats, start_date, end_date, churn_flag, ... |
| feature_usage.csv | subscription_id, feature_name, usage_count, usage_duration_secs, error_count |
| support_tickets.csv | ticket_id, account_id, resolution_time_hours, satisfaction_score, escalation_flag |
| churn_events.csv | churn_event_id, account_id |

Column names must match exactly, since the app's data-prep pipeline
looks for these specific names. A different schema would require
adapting the `load_and_prepare()` function.

## Known limitations

- The model retrains on every fresh data upload rather than using a
  saved/versioned model — fine for a demonstration, not a production
  pattern
- Health scores can cluster near the extremes (0 or 1) rather than
  spreading smoothly, a known effect of SMOTE oversampling on a
  weak-signal dataset; probability calibration (Platt scaling or
  isotonic regression) would improve this
- Assumes the fixed five-table RavenStack schema; does not
  auto-detect or adapt to arbitrary data structures
