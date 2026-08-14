# ============================================================
# PM-FACING ACCOUNT HEALTH SCORING TOOL
# Predictive Health Scoring for Product-Led Expansion
# ============================================================
# Run locally with:  streamlit run health_scoring_app.py
# Deploy free at:     https://share.streamlit.io  (connect this
#                      file via a GitHub repo, no server needed)
#
# Expects five CSVs in the RavenStack schema:
#   accounts.csv, subscriptions.csv, feature_usage.csv,
#   support_tickets.csv, churn_events.csv
# A PM at another company can upload their own CSVs in this same
# shape and get scored results without touching any code.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
st.set_page_config(page_title="Account Health Scoring", layout="wide")

# ============================================================
# SIDEBAR - DATA INPUT
# ============================================================
st.sidebar.title("Data Input")
use_demo = st.sidebar.checkbox("Use bundled demo data (RavenStack)", value=True)

if use_demo:
    st.sidebar.info("Using demo data. Uncheck to upload your own CSVs.")
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

    required = [accounts_file, subs_file, usage_file, support_file, churn_file]
    if not all(required):
        st.title("Account Health Scoring Tool")
        st.warning("Upload all five CSVs in the sidebar to begin, or "
                   "check 'Use bundled demo data' to try it with the "
                   "sample dataset.")
        st.stop()

# ============================================================
# DATA LOADING & PREP (cached so it only reruns when inputs change)
# ============================================================
@st.cache_data
def load_and_prepare(accounts_f, subs_f, usage_f, support_f, churn_f):
    accounts = pd.read_csv(accounts_f)
    subscriptions = pd.read_csv(subs_f)
    feature_usage = pd.read_csv(usage_f)
    support = pd.read_csv(support_f)
    churn_events = pd.read_csv(churn_f)

    # --- Aggregate to subscription/account level ---
    usage_agg = feature_usage.groupby('subscription_id').agg(
        total_usage_count=('usage_count', 'sum'),
        total_usage_duration=('usage_duration_secs', 'sum'),
        total_errors=('error_count', 'sum'),
        unique_features_used=('feature_name', 'nunique'),
        avg_usage_per_session=('usage_count', 'mean')
    ).reset_index()

    support_agg = support.groupby('account_id').agg(
        total_tickets=('ticket_id', 'count'),
        avg_resolution_hours=('resolution_time_hours', 'mean'),
        avg_satisfaction=('satisfaction_score', 'mean'),
        escalation_count=('escalation_flag', 'sum')
    ).reset_index()

    churn_agg = churn_events.groupby('account_id').agg(
        churn_event_count=('churn_event_id', 'count')
    ).reset_index()

    sub_usage = subscriptions.merge(usage_agg, on='subscription_id', how='left')
    master = accounts.merge(sub_usage, on='account_id', how='left')
    master = master.merge(support_agg, on='account_id', how='left')
    master = master.merge(churn_agg, on='account_id', how='left')

    master.columns = master.columns.str.replace(r'_x$', '', regex=True) \
                                     .str.replace(r'_y$', '_sub', regex=True)
    if 'churn_flag' not in master.columns and 'churn_flag_sub' in master.columns:
        master['churn_flag'] = master['churn_flag_sub']
    master['churn_flag'] = master['churn_flag'].astype(int)

    df = master.copy()
    leak_cols = ['churn_event_count', 'account_id', 'subscription_id',
                 'account_name', 'signup_date', 'start_date', 'end_date',
                 'arr_amount']
    leak_cols = [c for c in leak_cols if c in df.columns]

    # Structural-zero and median imputation
    for col in ['total_usage_count', 'total_usage_duration', 'total_errors',
                'unique_features_used', 'avg_usage_per_session']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in ['avg_satisfaction', 'avg_resolution_hours']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Tenure engineering
    if 'signup_date' in master.columns:
        df['signup_date'] = pd.to_datetime(master['signup_date'], errors='coerce')
        df['start_date'] = pd.to_datetime(master['start_date'], errors='coerce')
        ref_date = df['start_date'].max()
        df['account_age_days'] = (ref_date - df['signup_date']).dt.days
        df['subscription_tenure_days'] = (ref_date - df['start_date']).dt.days
        for col in ['account_age_days', 'subscription_tenure_days']:
            df[col] = df[col].fillna(df[col].median())

    if 'active_subscription' not in df.columns and 'end_date' in master.columns:
        df['active_subscription'] = master['end_date'].isnull().astype(int)

    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = [c for c in df.select_dtypes(include='object').columns
                         if c not in leak_cols]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    feature_cols = [c for c in df_encoded.columns
                     if c not in leak_cols + ['churn_flag']]

    X = df_encoded[feature_cols]
    y = df_encoded['churn_flag']
    groups = master['account_id'] if 'account_id' in master.columns else None
    mrr_col = 'mrr_amount' if 'mrr_amount' in master.columns else None

    return master, X, y, groups, mrr_col


with st.spinner("Loading and preparing data..."):
    master, X, y, groups, mrr_col = load_and_prepare(
        accounts_file, subs_file, usage_file, support_file, churn_file
    )

# ============================================================
# MODEL TRAINING (cached)
# ============================================================
@st.cache_resource
def train_model(X, y, groups):
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
    else:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_res, y_train_res)

    from sklearn.metrics import roc_auc_score, average_precision_score
    y_proba_test = model.predict_proba(X_test)[:, 1]
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba_test),
        'pr_auc': average_precision_score(y_test, y_proba_test)
    }
    return model, metrics


with st.spinner("Training model..."):
    model, metrics = train_model(X, y, groups)

explainer = shap.TreeExplainer(model)
churn_proba = model.predict_proba(X)[:, 1]

scored = master.copy()
scored['health_score'] = 1 - churn_proba
scored['churn_probability'] = churn_proba

if mrr_col:
    health_med = scored['health_score'].median()
    mrr_med = scored[mrr_col].median()

    def quadrant(row):
        if row['churn_flag'] == 1:
            return 'Lost'
        hi_health = row['health_score'] >= health_med
        hi_mrr = row[mrr_col] >= mrr_med
        if hi_health:
            return 'Healthy'
        return 'Critical' if hi_mrr else 'At Risk'

    scored['quadrant'] = scored.apply(quadrant, axis=1)

# ============================================================
# MAIN UI
# ============================================================
st.title("Account Health Scoring Dashboard")
st.caption("Predictive Health Scoring for Product-Led Expansion")

col1, col2, col3 = st.columns(3)
col1.metric("Accounts Scored", f"{len(scored):,}")
col2.metric("Model AUC-ROC", f"{metrics['roc_auc']:.3f}")
col3.metric("Model PR-AUC", f"{metrics['pr_auc']:.3f}")

tab1, tab2 = st.tabs(["Portfolio Overview", "Account Lookup"])

# --- TAB 1: Portfolio quadrant view ---
with tab1:
    st.subheader("Health Quadrants")

    if mrr_col:
        quad_counts = scored['quadrant'].value_counts()
        qcols = st.columns(4)
        colors = {'Healthy': '#2E86AB', 'At Risk': '#F4A261',
                  'Critical': '#E84855', 'Lost': '#6B6B6B'}
        for i, q in enumerate(['Healthy', 'At Risk', 'Critical', 'Lost']):
            qcols[i].metric(q, int(quad_counts.get(q, 0)))

        fig, ax = plt.subplots(figsize=(9, 6))
        for q, color in colors.items():
            subset = scored[scored['quadrant'] == q]
            ax.scatter(subset['health_score'], subset[mrr_col],
                       label=q, alpha=0.5, s=20, color=color)
        ax.axvline(health_med, color='grey', linestyle='--', alpha=0.5)
        ax.axhline(mrr_med, color='grey', linestyle='--', alpha=0.5)
        ax.set_xlabel('Health Score (1 - churn probability)')
        ax.set_ylabel('MRR (USD)')
        ax.legend()
        st.pyplot(fig)

        st.subheader("Critical Accounts (highest priority for outreach)")
        critical = scored[scored['quadrant'] == 'Critical'].sort_values(
            mrr_col, ascending=False
        )
        display_cols = [c for c in ['account_id', 'health_score', mrr_col,
                                     'quadrant'] if c in critical.columns]
        st.dataframe(critical[display_cols].head(20), use_container_width=True)
    else:
        st.warning("mrr_amount column not found - quadrant view needs MRR data.")

# --- TAB 2: Per-account lookup ---
with tab2:
    st.subheader("Look up an account")

    account_ids = master['account_id'].unique().tolist() if 'account_id' in master.columns else []
    selected = st.selectbox("Select account_id", account_ids)

    if selected:
        idx = master.index[master['account_id'] == selected][0]
        row = scored.loc[idx]

        c1, c2, c3 = st.columns(3)
        c1.metric("Health Score", f"{row['health_score']:.3f}")
        c2.metric("Churn Probability", f"{row['churn_probability']:.3f}")
        if mrr_col:
            c3.metric("MRR", f"${row[mrr_col]:,.0f}")
        st.write(f"**Quadrant:** {row.get('quadrant', 'N/A')}")

        # Live SHAP explanation for this account
        account_features = X.iloc[[idx]]
        exp = explainer(account_features)
        vals = exp.values[0]
        if np.ndim(vals) > 1:
            vals = vals[:, 1]

        contrib = pd.Series(vals, index=X.columns)

        # Only show categorical dummies that are TRUE (=1) for this account,
        # e.g. show "country_UK" not "country_US" for a UK-based account.
        # Numeric features are always kept.
        account_row = account_features.iloc[0]
        keep = [f for f in contrib.index
                if account_row[f] != 0 or '_' not in f]
        contrib = contrib[keep].sort_values(key=abs, ascending=False).head(10)

        st.subheader("Top factors driving this account's risk score")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_bar = ['#E84855' if v > 0 else '#2E86AB' for v in contrib.values]
        ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors_bar[::-1])
        ax.set_xlabel('SHAP value (positive = increases churn risk)')
        ax.axvline(0, color='black', linewidth=0.8)
        st.pyplot(fig)

        st.caption("Red bars increase churn risk, blue bars decrease it. "
                   "Magnitude shows the strength of each factor's influence "
                   "on this specific account's prediction.")
