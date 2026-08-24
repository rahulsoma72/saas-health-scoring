# ============================================================
# PM-FACING ACCOUNT HEALTH & EXPANSION PRIORITISATION TOOL
# Account-level, temporally-windowed churn risk and
# expansion opportunity assessment with SHAP-based
# churn interpretability.
#
# This tool is a DECISION-SUPPORT system, not an automated
# churn-detection or expansion-decision system. Predictions
# should be combined with PM/Customer Success judgement.
#
# Run locally:
#     streamlit run health_scoring_app.py
# ============================================================

import os
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Account Health & Expansion Prioritisation",
    layout="wide"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FINAL_FEATURES = [
    "account_age_days",
    "subscription_tenure_days",
    "days_since_latest_subscription_start"
]

EXPANSION_FEATURES = [
    "account_age_days",
    "subscription_tenure_days",
    "days_since_latest_subscription_start"
]

FROZEN_MODEL_PATH = "ravenstack_frozen_model.joblib"
FROZEN_METADATA_PATH = "ravenstack_frozen_model_metadata.json"

EXPANSION_MODEL_PATH = "ravenstack_expansion_model.joblib"
EXPANSION_METADATA_PATH = "ravenstack_expansion_model_metadata.json"


# ============================================================
# SIDEBAR - DATA INPUT AND PARAMETERS
# ============================================================

st.sidebar.title("Data & Settings")

use_demo = st.sidebar.checkbox(
    "Use bundled demo data (RavenStack)",
    value=True
)

if use_demo:

    accounts_file = "ravenstack_accounts.csv"
    subs_file = "ravenstack_subscriptions.csv"
    usage_file = "ravenstack_feature_usage.csv"
    support_file = "ravenstack_support_tickets.csv"
    churn_file = "ravenstack_churn_events.csv"

else:

    accounts_file = st.sidebar.file_uploader(
        "accounts.csv",
        type="csv"
    )

    subs_file = st.sidebar.file_uploader(
        "subscriptions.csv",
        type="csv"
    )

    usage_file = st.sidebar.file_uploader(
        "feature_usage.csv",
        type="csv"
    )

    support_file = st.sidebar.file_uploader(
        "support_tickets.csv",
        type="csv"
    )

    churn_file = st.sidebar.file_uploader(
        "churn_events.csv",
        type="csv"
    )

    if not all([
        accounts_file,
        subs_file,
        usage_file,
        support_file,
        churn_file
    ]):

        st.title(
            "Account Health & Expansion Prioritisation Tool"
        )

        st.warning(
            "Upload all five CSVs in the sidebar, or check "
            "'Use bundled demo data'."
        )

        st.stop()


st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Window")

cutoff_date = st.sidebar.date_input(
    "Prediction cutoff date",
    value=pd.Timestamp("2024-09-30"),
    help=(
        "Default settings match the dissertation modelling "
        "configuration: 30 September 2024 cutoff with a "
        "90-day historical lookback. Changing the cutoff "
        "creates a different historical account snapshot and "
        "recalculates the model inputs for that date. The "
        "same frozen churn and expansion models are applied; "
        "neither model is retrained. Prediction dates are "
        "available only where the loaded data provide "
        "sufficient historical information."
    )
)

cutoff_date = pd.Timestamp(cutoff_date)

lookback_days = st.sidebar.number_input(
    "Lookback window (days)",
    min_value=30,
    max_value=365,
    value=90
)


# ============================================================
# DATA CONSTRUCTION
# ============================================================

@st.cache_data
def build_account_dataset(
    accounts_f,
    subs_f,
    usage_f,
    support_f,
    churn_f,
    cutoff,
    lookback
):

    accounts = pd.read_csv(
        accounts_f,
        parse_dates=["signup_date"]
    )

    subscriptions = pd.read_csv(
        subs_f,
        parse_dates=["start_date", "end_date"]
    )

    feature_usage = pd.read_csv(
        usage_f,
        parse_dates=["usage_date"]
    )

    support = pd.read_csv(
        support_f,
        parse_dates=["submitted_at", "closed_at"]
    )

    churn_events = pd.read_csv(
        churn_f,
        parse_dates=["churn_date"]
    )

    lookback_start = (
        cutoff -
        pd.Timedelta(days=lookback)
    )

    # --------------------------------------------------------
    # Active accounts at cutoff
    # --------------------------------------------------------

    active_subs = subscriptions[
        (subscriptions["start_date"] <= cutoff) &
        (
            subscriptions["end_date"].isna() |
            (subscriptions["end_date"] >= cutoff)
        )
    ].copy()

    active_account_ids = (
        active_subs["account_id"]
        .dropna()
        .unique()
    )

    # --------------------------------------------------------
    # Lifecycle dates
    # --------------------------------------------------------

    sub_dates = (
        active_subs
        .groupby("account_id")
        .agg(
            first_subscription_date=(
                "start_date",
                "min"
            ),
            latest_subscription_start=(
                "start_date",
                "max"
            )
        )
        .reset_index()
    )

    sub_dates["subscription_tenure_days"] = (
        cutoff -
        sub_dates["first_subscription_date"]
    ).dt.days

    sub_dates["days_since_latest_subscription_start"] = (
        cutoff -
        sub_dates["latest_subscription_start"]
    ).dt.days

    # --------------------------------------------------------
    # Master account table
    # --------------------------------------------------------

    master = accounts[
        accounts["account_id"].isin(
            active_account_ids
        )
    ].copy()

    master["account_age_days"] = (
        cutoff -
        master["signup_date"]
    ).dt.days

    master = master.merge(
        sub_dates[
            [
                "account_id",
                "subscription_tenure_days",
                "days_since_latest_subscription_start"
            ]
        ],
        on="account_id",
        how="left"
    )

    # --------------------------------------------------------
    # Future churn target
    #
    # Used only where the selected cutoff allows a
    # look-ahead within the available data. It is NOT
    # used as a feature.
    # --------------------------------------------------------

    prediction_end = (
        cutoff +
        pd.Timedelta(days=lookback)
    )

    future_churn = churn_events[
        (churn_events["churn_date"] > cutoff) &
        (churn_events["churn_date"] <= prediction_end) &
        (~churn_events["is_reactivation"]) &
        (
            churn_events["account_id"]
            .isin(active_account_ids)
        )
    ].copy()

    future_target = (
        future_churn
        .groupby("account_id")
        .size()
        .reset_index(
            name="future_churn_event_count"
        )
    )

    future_target["target_future_churn"] = 1

    master = master.merge(
        future_target[
            [
                "account_id",
                "target_future_churn"
            ]
        ],
        on="account_id",
        how="left"
    )

    master["target_future_churn"] = (
        master["target_future_churn"]
        .fillna(0)
        .astype(int)
    )

    return master, active_account_ids


with st.spinner(
    "Building account-level dataset for the selected cutoff..."
):

    master, active_ids = build_account_dataset(
        accounts_file,
        subs_file,
        usage_file,
        support_file,
        churn_file,
        cutoff_date,
        lookback_days
    )


# ============================================================
# LOAD FROZEN CHURN MODEL
# ============================================================

if not os.path.exists(FROZEN_MODEL_PATH):

    st.error(
        f"Frozen dissertation model file "
        f"'{FROZEN_MODEL_PATH}' was not found. "
        "The final version of this tool requires the "
        "evaluated frozen model and does not retrain a "
        "replacement model."
    )

    st.stop()


if not os.path.exists(FROZEN_METADATA_PATH):

    st.error(
        f"Frozen model metadata file "
        f"'{FROZEN_METADATA_PATH}' was not found."
    )

    st.stop()


try:

    raw_classifier = joblib.load(
        FROZEN_MODEL_PATH
    )

    with open(
        FROZEN_METADATA_PATH,
        "r"
    ) as f:

        model_metadata = json.load(f)

except Exception as e:

    st.error(
        "The frozen dissertation churn model could not "
        f"be loaded: {e}"
    )

    st.stop()


imputer_medians = model_metadata[
    "imputer_medians"
]


# ============================================================
# LOAD FROZEN EXPANSION MODEL
# ============================================================

if not os.path.exists(EXPANSION_MODEL_PATH):

    st.error(
        f"Frozen expansion model file "
        f"'{EXPANSION_MODEL_PATH}' was not found. "
        "Add the final expansion model to the deployment."
    )

    st.stop()


if not os.path.exists(EXPANSION_METADATA_PATH):

    st.error(
        f"Expansion model metadata file "
        f"'{EXPANSION_METADATA_PATH}' was not found."
    )

    st.stop()


try:

    expansion_model = joblib.load(
        EXPANSION_MODEL_PATH
    )

    with open(
        EXPANSION_METADATA_PATH,
        "r"
    ) as f:

        expansion_metadata = json.load(f)

except Exception as e:

    st.error(
        "The frozen expansion model could not "
        f"be loaded: {e}"
    )

    st.stop()


# ============================================================
# CHURN PREDICTIONS
# ============================================================

X_all = master[
    FINAL_FEATURES
].copy()

for col in FINAL_FEATURES:

    X_all[col] = (
        X_all[col]
        .fillna(imputer_medians[col])
    )


model = raw_classifier

churn_proba = (
    model
    .predict_proba(X_all)[:, 1]
)


scored = master.copy()

scored["churn_probability"] = (
    churn_proba
)

scored["model_derived_health_score"] = (
    1 -
    churn_proba
) * 100


# ============================================================
# CHURN RISK BANDS
# ============================================================

def risk_band(p):

    if p >= 0.40:
        return "Critical"

    elif p >= 0.30:
        return "High"

    elif p >= 0.20:
        return "Moderate"

    else:
        return "Low"


scored["risk_band"] = (
    scored["churn_probability"]
    .apply(risk_band)
)


# ============================================================
# EXPANSION PREDICTIONS
# ============================================================

X_expansion = scored[
    EXPANSION_FEATURES
].copy()


expansion_proba = (
    expansion_model
    .predict_proba(X_expansion)[:, 1]
)


scored["expansion_probability"] = (
    expansion_proba
)


# ============================================================
# EXPANSION OPPORTUNITY BANDS
# ============================================================

def expansion_band(p):

    if p >= 0.56:
        return "High"

    elif p >= 0.42:
        return "Moderate"

    else:
        return "Low"


scored["expansion_opportunity"] = (
    scored["expansion_probability"]
    .apply(expansion_band)
)


# ============================================================
# COMBINED PM ACTION
# ============================================================

def suggested_action(row):

    risk = row["risk_band"]
    opportunity = row["expansion_opportunity"]

    if risk == "Critical" and opportunity == "High":
        return (
            "🔴 Prioritise retention intervention and defer "
            "expansion outreach until account health is stabilised."
        )

    if risk == "High" and opportunity == "High":
        return (
            "🔴 Address retention risk first; reassess expansion "
            "once account health improves."
        )

    if risk == "Moderate" and opportunity == "High":
        return (
            "🟠 Review account health and engagement before "
            "initiating expansion outreach."
        )

    if risk == "Low" and opportunity == "High":
        return (
            "🟢 Prioritise a targeted expansion conversation while "
            "confirming continued account health."
        )

    if risk in ["Critical", "High"]:
        return (
            "🔴 Prioritise proactive retention engagement and "
            "investigate the main risk signals."
        )

    if opportunity == "Moderate":
        return (
            "🟡 Assess expansion fit alongside account health "
            "during routine account review."
        )

    if risk == "Moderate":
        return (
            "🟡 Monitor account health and engagement for "
            "emerging retention concerns."
        )

    return (
        "🟢 Continue routine monitoring and account engagement."
    )


scored["suggested_action"] = (
    scored
    .apply(
        suggested_action,
        axis=1
    )
)


# ============================================================
# SHAP — CHURN MODEL ONLY
# ============================================================

explainer = shap.TreeExplainer(
    model
)

raw_shap = explainer.shap_values(
    X_all
)


if isinstance(raw_shap, list):

    shap_vals = raw_shap[1]

elif (
    isinstance(raw_shap, np.ndarray)
    and raw_shap.ndim == 3
):

    shap_vals = raw_shap[:, :, 1]

else:

    shap_vals = raw_shap


# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    """
    <style>
    .suggested-action-card {
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.9rem;
        margin-bottom: 0.8rem;
    }

    .suggested-action-label {
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .suggested-action-text {
        line-height: 1.45;
    }

    .shap-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 0.3rem;
        margin-bottom: 0.2rem;
    }

    /* Hide Streamlit heading anchor/link icons while retaining headings. */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
    h1 button, h2 button, h3 button, h4 button, h5 button, h6 button {
        display: none !important;
        visibility: hidden !important;
    }

    .shap-info {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.05rem;
        height: 1.05rem;
        margin-left: 0.35rem;
        border: 1px solid rgba(180, 180, 180, 0.65);
        border-radius: 50%;
        font-size: 0.72rem;
        font-weight: 700;
        cursor: help;
        opacity: 0.8;
        vertical-align: 0.08rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(
    "Account Health & Expansion Prioritisation"
)


# ------------------------------------------------------------
# Frozen model status
# ------------------------------------------------------------

st.success(
    f"✅ Using the frozen dissertation churn model "
    f"({model_metadata['model_type']}, trained on "
    f"{model_metadata['trained_on_n_accounts']} accounts, "
    f"test ROC-AUC = "
    f"{model_metadata['test_roc_auc']:.4f}). "
    f"Predictions below come from this evaluated model."
)


# Decision-support guidance is available through compact hover icons in each tab.


# ------------------------------------------------------------
# Model status / expansion information
# ------------------------------------------------------------

with st.expander(
    "About the models used in this tool"
):

    st.markdown(
        f"""
**Retention model**

- Model: {model_metadata['model_type']}
- Training accounts: {model_metadata['trained_on_n_accounts']}
- Test ROC-AUC: {model_metadata['test_roc_auc']:.4f}
- Output: churn probability and model-derived health score

**Expansion model**

- Model: {expansion_metadata['model_type']}
- Target: future account upgrade within 90 days
- Training accounts: {expansion_metadata['training_accounts']}
- Fresh holdout ROC-AUC: {expansion_metadata['fresh_holdout_roc_auc']:.4f}
- Output: expansion probability


"""
    )


# ============================================================
# PORTFOLIO METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Active accounts",
    f"{len(scored):,}"
)

col2.metric(
    "Critical churn risk",
    int(
        (
            scored["risk_band"] ==
            "Critical"
        ).sum()
    ),
    help=(
        "Critical churn-risk band: predicted churn probability "
        "of 40% or higher."
    )
)

col3.metric(
    "High churn risk",
    int(
        (
            scored["risk_band"] ==
            "High"
        ).sum()
    ),
    help=(
        "High churn-risk band: predicted churn probability "
        "from 30% to below 40%."
    )
)

col4.metric(
    "Avg. expansion probability",
    f"{scored['expansion_probability'].mean()*100:.1f}%",
    help=(
        "Average predicted probability of a future account "
        "upgrade within the model's 90-day expansion horizon."
    )
)



# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Portfolio Overview",
        "Account Investigation",
        "Expansion Opportunities"
    ]
)


# ============================================================
# TAB 1 — PORTFOLIO OVERVIEW
# ============================================================

with tab1:

    st.subheader(
        "Which accounts require attention?"
    )
    band_order = [
        "Critical",
        "High",
        "Moderate",
        "Low"
    ]

    band_counts = (
        scored["risk_band"]
        .value_counts()
        .reindex(
            band_order
        )
        .fillna(0)
    )


    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    colors = {
        "Critical": "#E84855",
        "High": "#F4A261",
        "Moderate": "#F9C74F",
        "Low": "#2E86AB"
    }

    ax.bar(
        band_counts.index,
        band_counts.values,
        color=[
            colors[b]
            for b in band_counts.index
        ]
    )

    ax.set_ylabel(
        "Number of accounts"
    )

    ax.set_title(
        "Accounts by Risk Band"
    )

    st.pyplot(
        fig
    )


    st.subheader(
        "Accounts ranked by predicted risk"
    )

    st.caption(
        "Ranking-based prioritisation lets a PM review "
        "a fixed number of highest-risk accounts rather "
        "than relying only on a fixed probability threshold."
    )


    display_cols = [
        "account_id",
        "churn_probability",
        "model_derived_health_score",
        "risk_band",
        "expansion_probability",
        "expansion_opportunity",
        "suggested_action"
    ] + FINAL_FEATURES


    display_cols = [
        c
        for c in display_cols
        if c in scored.columns
    ]


    ranked = (
        scored[display_cols]
        .sort_values(
            "churn_probability",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    ranked["churn_probability"] = (
        ranked["churn_probability"] *
        100
    ).round(1)


    ranked["model_derived_health_score"] = (
        ranked["model_derived_health_score"]
        .round(1)
    )


    ranked["expansion_probability"] = (
        ranked["expansion_probability"] *
        100
    ).round(1)


    ranked = ranked.rename(
        columns={
            "churn_probability":
                "churn_probability_pct",

            "model_derived_health_score":
                "health_score_0_100",

            "expansion_probability":
                "expansion_probability_pct"
        }
    )


    top_n = st.slider(
        "Show top N highest-risk accounts",
        5,
        min(
            100,
            len(ranked)
        ),
        20
    )


    st.dataframe(
        ranked.head(top_n),
        use_container_width=True
    )







# ============================================================
# TAB 2 — ACCOUNT INVESTIGATION
# ============================================================

with tab2:

    st.subheader(
        "Account-level investigation"
    )

    account_ids = (
        master["account_id"]
        .unique()
        .tolist()
    )


    selected = st.selectbox(
        "Select an account",
        account_ids
    )


    if selected:

        idx = (
            master.index[
                master["account_id"] ==
                selected
            ][0]
        )

        row = scored.loc[idx]


        # ----------------------------------------------------
        # RETENTION + EXPANSION SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "### Account decision-support summary"
        )


        # A small central gap keeps Retention and Expansion visually distinct.
        left, gap, right = st.columns([1, 0.20, 1])

        with left:

            st.markdown(
                "#### Retention"
            )

            c1, c2, c3 = st.columns([1.0, 1.35, 1.0])

            c1.metric(
                "Churn probability",
                f"{row['churn_probability']*100:.1f}%"
            )

            c2.metric(
                "Health score",
                f"{row['model_derived_health_score']:.1f} / 100"
            )

            c3.metric(
                "Risk band",
                row["risk_band"]
            )

        with gap:
            st.empty()

        with right:

            st.markdown(
                "#### Expansion"
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Expansion probability",
                f"{row['expansion_probability']*100:.1f}%"
            )

            c2.metric(
                "Opportunity",
                row["expansion_opportunity"]
            )

        st.markdown(
            f'**Suggested action:** {row["suggested_action"]}'
        )



        st.markdown("---")


        # ----------------------------------------------------
        # MODEL INPUTS
        # ----------------------------------------------------

        st.write(
            "**Account lifecycle information used by "
            "the models:**"
        )


        info_cols = st.columns(3)


        info_cols[0].metric(
            "Account age (days)",
            f"{row['account_age_days']:.0f}"
        )


        info_cols[1].metric(
            "Subscription tenure (days)",
            f"{row['subscription_tenure_days']:.0f}"
        )


        info_cols[2].metric(
            "Days since latest subscription start",
            f"{row['days_since_latest_subscription_start']:.0f}"
        )

        # ----------------------------------------------------
        # HEALTH SCORE EXPLANATION
        # ----------------------------------------------------

        with st.expander(
            "How the health score is calculated"
        ):

            st.write(
                "The health score is a model-derived score "
                "calculated as 100 minus the predicted churn "
                "probability, expressed on a 0–100 scale. "
                "It is not an independently validated health "
                "metric."
            )
        
        st.markdown("---")





        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="shap-title">
                Why: factors contributing to this prediction
                <span
                    class="shap-info"
                    title="These values show each factor's contribution to this account's specific churn prediction, not a causal explanation. A factor increasing predicted risk does not mean changing it would prevent churn. Use this to guide investigation, not as a prescribed intervention."
                >i</span>
            </div>
            """,
            unsafe_allow_html=True
        )


        pos_in_array = (
            master.index.get_loc(idx)
        )


        contrib = pd.Series(
            shap_vals[pos_in_array],
            index=FINAL_FEATURES
        ).sort_values(
            key=abs,
            ascending=False
        )


        fig, ax = plt.subplots(
            figsize=(9, 4)
        )


        bar_colors = [
            "#E84855"
            if v > 0
            else "#2E86AB"
            for v in contrib.values
        ]


        ax.barh(
            contrib.index[::-1],
            contrib.values[::-1],
            color=bar_colors[::-1]
        )


        ax.axvline(
            0,
            color="black",
            linewidth=0.8
        )


        ax.set_xlabel(
            "SHAP value "
            "(positive = increases predicted risk)"
        )


        st.pyplot(
            fig
        )








# ============================================================
# TAB 3 — EXPANSION OPPORTUNITIES
# ============================================================

with tab3:

    st.subheader(
        "Which accounts show expansion opportunity?"
    )
    with st.expander(
        "How expansion probability is calculated"
    ):
        st.write(
            "Expansion probability is produced by the frozen "
            "Logistic Regression model using account age, "
            "subscription tenure, and days since the latest "
            "subscription start. It represents the model's "
            "statistical signal for a future account upgrade "
            "within 90 days. It is a prioritisation signal, "
            "not a guarantee of expansion."
        )


    expansion_order = [
        "High",
        "Moderate",
        "Low"
    ]


    expansion_counts = (
        scored["expansion_opportunity"]
        .value_counts()
        .reindex(
            expansion_order
        )
        .fillna(0)
    )


    fig, ax = plt.subplots(
        figsize=(8, 4)
    )


    expansion_colors = {
        "High": "#2E8B57",
        "Moderate": "#F9C74F",
        "Low": "#7A7A7A"
    }


    ax.bar(
        expansion_counts.index,
        expansion_counts.values,
        color=[
            expansion_colors[b]
            for b in expansion_counts.index
        ]
    )


    ax.set_ylabel(
        "Number of accounts"
    )


    ax.set_title(
        "Accounts by Expansion Opportunity"
    )


    st.pyplot(
        fig
    )


    st.subheader(
        "Highest expansion opportunities"
    )


    expansion_display = scored[
        [
            "account_id",
            "expansion_probability",
            "expansion_opportunity",
            "churn_probability",
            "model_derived_health_score",
            "risk_band",
            "suggested_action"
        ] + FINAL_FEATURES
    ].copy()


    expansion_display = (
        expansion_display
        .sort_values(
            "expansion_probability",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    expansion_display["expansion_probability"] = (
        expansion_display[
            "expansion_probability"
        ] *
        100
    ).round(1)


    expansion_display["churn_probability"] = (
        expansion_display[
            "churn_probability"
        ] *
        100
    ).round(1)


    expansion_display[
        "model_derived_health_score"
    ] = (
        expansion_display[
            "model_derived_health_score"
        ]
        .round(1)
    )


    expansion_display = (
        expansion_display
        .rename(
            columns={
                "expansion_probability":
                    "expansion_probability_pct",

                "churn_probability":
                    "churn_probability_pct",

                "model_derived_health_score":
                    "health_score_0_100"
            }
        )
    )


    top_expansion_n = st.slider(
        "Show top N expansion opportunities",
        5,
        min(
            100,
            len(expansion_display)
        ),
        20,
        key="expansion_top_n"
    )


    st.dataframe(
        expansion_display.head(
            top_expansion_n
        ),
        use_container_width=True
    )


    st.markdown("---")


    st.subheader(
        "PM interpretation"
    )


    st.write(
        """
Accounts with high expansion probability should not
automatically be treated as immediate upsell targets.
The expansion signal should be considered together with
retention health. In particular, an account showing both
high churn risk and high expansion probability should
generally be reviewed for retention or account-health
issues before expansion outreach.
"""
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.caption(
    "Decision-support prototype developed for the RavenStack dissertation. "
    "Churn and expansion models are separate frozen predictive models. "
    "Model outputs should support, not replace, Product Management and "
    "Customer Success judgement."
)
