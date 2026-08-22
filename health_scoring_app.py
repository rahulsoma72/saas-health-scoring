# ============================================================
# PM-FACING ACCOUNT HEALTH & RISK-PRIORITISATION TOOL
# Account-level, temporally-windowed churn risk assessment
# with SHAP-based interpretability.
#
# This tool is a DECISION-SUPPORT system, not an automated
# churn-detection system. Predictions should be combined with
# PM/Customer Success judgement, not treated as certainties.
#
# Run locally:   streamlit run pm_health_tool.py
# Deploy free:   push to GitHub, then deploy via share.streamlit.io
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

st.set_page_config(
    page_title="Account Health & Risk Prioritisation",
    layout="wide"
)

FINAL_FEATURES = [
    "account_age_days",
    "subscription_tenure_days",
    "days_since_latest_subscription_start"
]

# ============================================================
# SIDEBAR - DATA INPUT AND PARAMETERS
# ============================================================
st.sidebar.title("Data & Settings")

use_demo = st.sidebar.checkbox("Use bundled demo data (RavenStack)", value=True)

if use_demo:
    accounts_file = "ravenstack_accounts.csv"
    subs_file = "ravenstack_subscriptions.csv"
    usage_file = "ravenstack_feature_usage.csv"
    support_file = "ravenstack_support_tickets.csv"
    churn_file = "ravenstack_churn_events.csv"
else:
    accounts_file = st.sidebar.file_uploader("accounts.csv", type="csv")
    subs_file = st.sidebar.file_uploader("subscriptions.csv", type="csv")
    usage_file = st.sidebar.file_uploader("feature_usage.csv", type="csv")
    support_file = st.sidebar.file_uploader("support_tickets.csv", type="csv")
    churn_file = st.sidebar.file_uploader("churn_events.csv", type="csv")

    if not all([accounts_file, subs_file, usage_file, support_file, churn_file]):
        st.title("Account Health & Risk Prioritisation Tool")
        st.warning("Upload all five CSVs in the sidebar, or check "
                   "'Use bundled demo data'.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Window")

cutoff_date = st.sidebar.date_input(
    "Prediction cutoff date",
    value=pd.Timestamp("2024-09-30"),
    help=(
        "About prediction settings: Default settings match the "
        "dissertation's final modelling configuration - 30 September "
        "2024 cutoff with a 90-day historical lookback. Changing the "
        "cutoff creates a different historical account snapshot and "
        "recalculates the model inputs for that date. The same frozen "
        "Random Forest is applied; the model is not retrained. "
        "Prediction dates are available only where the loaded data "
        "provide sufficient historical information."
    )
)
cutoff_date = pd.Timestamp(cutoff_date)

lookback_days = st.sidebar.number_input(
    "Lookback window (days)", min_value=30, max_value=365, value=90
)



# ============================================================
# DATA CONSTRUCTION (cached, matches the dissertation pipeline)
# ============================================================
@st.cache_data
def build_account_dataset(accounts_f, subs_f, usage_f, support_f, churn_f,
                            cutoff, lookback):
    accounts = pd.read_csv(accounts_f, parse_dates=["signup_date"])
    subscriptions = pd.read_csv(subs_f, parse_dates=["start_date", "end_date"])
    feature_usage = pd.read_csv(usage_f, parse_dates=["usage_date"])
    support = pd.read_csv(support_f, parse_dates=["submitted_at", "closed_at"])
    churn_events = pd.read_csv(churn_f, parse_dates=["churn_date"])

    lookback_start = cutoff - pd.Timedelta(days=lookback)

    # --- Active accounts at cutoff ---
    active_subs = subscriptions[
        (subscriptions["start_date"] <= cutoff) &
        (subscriptions["end_date"].isna() | (subscriptions["end_date"] >= cutoff))
    ].copy()
    active_account_ids = active_subs["account_id"].dropna().unique()

    # --- Lifecycle dates ---
    sub_dates = active_subs.groupby("account_id").agg(
        first_subscription_date=("start_date", "min"),
        latest_subscription_start=("start_date", "max")
    ).reset_index()
    sub_dates["subscription_tenure_days"] = (
        cutoff - sub_dates["first_subscription_date"]
    ).dt.days
    sub_dates["days_since_latest_subscription_start"] = (
        cutoff - sub_dates["latest_subscription_start"]
    ).dt.days

    # --- Build master account table ---
    master = accounts[accounts["account_id"].isin(active_account_ids)].copy()
    master["account_age_days"] = (cutoff - master["signup_date"]).dt.days
    master = master.merge(
        sub_dates[["account_id", "subscription_tenure_days",
                   "days_since_latest_subscription_start"]],
        on="account_id", how="left"
    )

    # --- Future churn target (only computable if cutoff allows a look-ahead
    #     within the available data; used for historical evaluation display) ---
    prediction_end = cutoff + pd.Timedelta(days=lookback)
    future_churn = churn_events[
        (churn_events["churn_date"] > cutoff) &
        (churn_events["churn_date"] <= prediction_end) &
        (~churn_events["is_reactivation"]) &
        (churn_events["account_id"].isin(active_account_ids))
    ].copy()
    future_target = future_churn.groupby("account_id").size().reset_index(
        name="future_churn_event_count"
    )
    future_target["target_future_churn"] = 1
    master = master.merge(
        future_target[["account_id", "target_future_churn"]],
        on="account_id", how="left"
    )
    master["target_future_churn"] = master["target_future_churn"].fillna(0).astype(int)

    return master, active_account_ids


with st.spinner("Building account-level dataset for the selected cutoff..."):
    master, active_ids = build_account_dataset(
        accounts_file, subs_file, usage_file, support_file, churn_file,
        cutoff_date, lookback_days
    )

# ============================================================
# MODEL: LOAD THE ACTUAL FROZEN DISSERTATION MODEL
# This is the exact model trained on 332 training accounts and
# evaluated once on the untouched 83-account test set. It is
# loaded here, not refit, so predictions genuinely come from
# the model reported and evaluated in the dissertation.
# ============================================================
import os
import joblib
import json

FROZEN_MODEL_PATH = "ravenstack_frozen_model.joblib"
FROZEN_METADATA_PATH = "ravenstack_frozen_model_metadata.json"

frozen_model_loaded = os.path.exists(FROZEN_MODEL_PATH)

if frozen_model_loaded:
    raw_classifier = joblib.load(FROZEN_MODEL_PATH)
    with open(FROZEN_METADATA_PATH) as f:
        model_metadata = json.load(f)
    imputer_medians = model_metadata["imputer_medians"]
else:
    st.warning(
        "⚠️ Frozen dissertation model file "
        f"('{FROZEN_MODEL_PATH}') not found in this deployment. "
        "Falling back to fitting a model on the currently loaded "
        "data — **this is NOT the frozen model evaluated in the "
        "dissertation** and should not be described as such. "
        "To use the actual frozen model, run save_frozen_model.py "
        "in Colab after your main pipeline and include both output "
        "files alongside this app."
    )

    @st.cache_resource
    def fit_fallback_model(master_df):
        X = master_df[FINAL_FEATURES]
        y = master_df["target_future_churn"]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=3,
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ])
        pipe.fit(X, y)
        return pipe

    fallback_pipe = fit_fallback_model(master)
    raw_classifier = fallback_pipe.named_steps["model"]
    imputer_medians = dict(zip(
        FINAL_FEATURES, fallback_pipe.named_steps["imputer"].statistics_.tolist()
    ))
    model_metadata = None

# Apply the same median-fill manually - avoids unpickling any
# SimpleImputer object, sidestepping sklearn version mismatches.
X_all = master[FINAL_FEATURES].copy()
for col in FINAL_FEATURES:
    X_all[col] = X_all[col].fillna(imputer_medians[col])

model = raw_classifier
churn_proba = model.predict_proba(X_all)[:, 1]

scored = master.copy()
scored["churn_probability"] = churn_proba
# This is a model-derived health score: simply the inverse of the
# predicted churn probability, expressed on a 0-100 scale for
# readability. It is not an independently validated health metric -
# stated explicitly here and in the UI caption below.
scored["model_derived_health_score"] = (1 - churn_proba) * 100

def risk_band(p):
    if p >= 0.40:
        return "Critical"
    elif p >= 0.30:
        return "High"
    elif p >= 0.20:
        return "Moderate"
    else:
        return "Low"

scored["risk_band"] = scored["churn_probability"].apply(risk_band)

action_map = {
    "Critical": "🔴 Prioritise for immediate account review",
    "High": "🟠 Prioritise for proactive engagement",
    "Moderate": "🟡 Monitor account health and engagement",
    "Low": "🟢 Continue routine monitoring"
}
scored["recommended_action"] = scored["risk_band"].map(action_map)

explainer = shap.TreeExplainer(model)
raw_shap = explainer.shap_values(X_all)

if isinstance(raw_shap, list):
    shap_vals = raw_shap[1]
elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
    shap_vals = raw_shap[:, :, 1]
else:
    shap_vals = raw_shap

# ============================================================
# MAIN UI
# ============================================================
st.title("Account Health & Risk Prioritisation")

if frozen_model_loaded:
    st.success(
        f"✅ Using the frozen dissertation model ({model_metadata['model_type']}, "
        f"trained on {model_metadata['trained_on_n_accounts']} accounts, "
        f"test ROC-AUC = {model_metadata['test_roc_auc']:.4f}). "
        f"Predictions below come from this exact evaluated model."
    )

st.info(
    "**This tool is a decision-support system, not an automated "
    "churn-detection system.** Predicted risk reflects modest, "
    "interpretable statistical signal from account lifecycle "
    "information. A high or low prediction is not proof an account "
    "will or will not churn — use this alongside your own knowledge "
    "of each account, not as a replacement for it."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active accounts", f"{len(scored):,}")
col2.metric("Critical", int((scored["risk_band"] == "Critical").sum()))
col3.metric("High", int((scored["risk_band"] == "High").sum()))
col4.metric("Avg. predicted risk", f"{scored['churn_probability'].mean()*100:.1f}%")

tab1, tab2 = st.tabs(["Portfolio Overview", "Account Investigation"])

# --- TAB 1: Portfolio-level view ---
with tab1:
    st.subheader("Which accounts require attention?")

    band_order = ["Critical", "High", "Moderate", "Low"]
    band_counts = scored["risk_band"].value_counts().reindex(band_order).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"Critical": "#E84855", "High": "#F4A261",
              "Moderate": "#F9C74F", "Low": "#2E86AB"}
    ax.bar(band_counts.index, band_counts.values,
           color=[colors[b] for b in band_counts.index])
    ax.set_ylabel("Number of accounts")
    ax.set_title("Accounts by Risk Band")
    st.pyplot(fig)

    st.subheader("Accounts ranked by predicted risk")
    st.caption(
        "Ranking-based prioritisation lets a PM review a fixed number "
        "of highest-risk accounts, rather than relying only on a fixed "
        "probability threshold."
    )

    display_cols = ["account_id", "churn_probability", "model_derived_health_score",
                     "risk_band", "recommended_action"] + FINAL_FEATURES
    display_cols = [c for c in display_cols if c in scored.columns]

    ranked = scored[display_cols].sort_values(
        "churn_probability", ascending=False
    ).reset_index(drop=True)
    ranked["churn_probability"] = (ranked["churn_probability"] * 100).round(1)
    ranked["model_derived_health_score"] = ranked["model_derived_health_score"].round(1)
    ranked = ranked.rename(columns={
        "churn_probability": "churn_probability_pct",
        "model_derived_health_score": "health_score_0_100"
    })

    top_n = st.slider("Show top N highest-risk accounts", 5, min(100, len(ranked)), 20)
    st.dataframe(ranked.head(top_n), use_container_width=True)

# --- TAB 2: Per-account investigation ---
with tab2:
    st.subheader("Why has this account been assigned its current risk level?")

    account_ids = master["account_id"].unique().tolist()
    selected = st.selectbox("Select an account", account_ids)

    if selected:
        idx = master.index[master["account_id"] == selected][0]
        row = scored.loc[idx]

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted churn probability", f"{row['churn_probability']*100:.1f}%")
        c2.metric("Health score", f"{row['model_derived_health_score']:.1f} / 100")
        c3.metric("Risk band", row["risk_band"])
        st.caption(
            "The health score is a model-derived score: simply "
            "100 minus the predicted churn probability, expressed on "
            "a 0-100 scale for readability. It is not an independently "
            "validated health metric."
        )

        st.write(f"**Suggested action:** {row['recommended_action']}")

        st.markdown("---")
        st.write("**Account lifecycle information used by the model:**")
        info_cols = st.columns(3)
        info_cols[0].metric("Account age (days)", f"{row['account_age_days']:.0f}")
        info_cols[1].metric("Subscription tenure (days)",
                             f"{row['subscription_tenure_days']:.0f}")
        info_cols[2].metric("Days since latest subscription start",
                             f"{row['days_since_latest_subscription_start']:.0f}")

        st.markdown("---")
        st.subheader("Why: factors contributing to this prediction")

        pos_in_array = master.index.get_loc(idx)
        contrib = pd.Series(
            shap_vals[pos_in_array], index=FINAL_FEATURES
        ).sort_values(key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(7, 3))
        bar_colors = ["#E84855" if v > 0 else "#2E86AB" for v in contrib.values]
        ax.barh(contrib.index[::-1], contrib.values[::-1], color=bar_colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (positive = increases predicted risk)")
        st.pyplot(fig)

        st.caption(
            "These values show each factor's contribution to this "
            "account's specific prediction, not a causal explanation. "
            "A factor increasing predicted risk does not mean changing "
            "it would prevent churn — use this to guide investigation, "
            "not as a prescribed intervention."
        )
