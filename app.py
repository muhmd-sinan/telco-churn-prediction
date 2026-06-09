import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Churn Analysis",
    page_icon=None,
    layout="wide"
)


# --------------------------------------------------
# Custom CSS — muted, professional colour scheme
# Slate background, charcoal text, amber accent
# --------------------------------------------------

st.markdown("""
<style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8f7f4;
        color: #1e1e1e;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e2329;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.9rem;
        padding: 6px 0;
    }

    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e4e0d8;
        border-radius: 6px;
        padding: 20px 24px;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e1e1e;
        line-height: 1;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 4px;
    }
    .metric-value.warn {
        color: #b45309;
    }

    /* Section headers */
    h2 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1e1e1e !important;
        border-bottom: 2px solid #e4e0d8;
        padding-bottom: 6px;
        margin-top: 28px !important;
    }

    /* Tables */
    .stDataFrame {
        border: 1px solid #e4e0d8 !important;
        border-radius: 6px !important;
    }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Remove extra top padding */
    .block-container {
        padding-top: 2rem;
    }

    /* Page title */
    .page-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e1e1e;
        margin-bottom: 2px;
    }
    .page-subtitle {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 28px;
    }

    /* Model badge */
    .badge {
        display: inline-block;
        background-color: #f0ece4;
        border: 1px solid #d1c9b8;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #5c4a1e;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load data and models (cached so they only load once)
# --------------------------------------------------

@st.cache_data
def load_data():
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    X_train = pd.read_csv('data/processed/X_train.csv')
    raw = pd.read_csv('data/raw/Telco-Customer-Churn.csv')

    X_test = X_test.fillna(0)
    X_train = X_train.fillna(0)

    return X_test, y_test, X_train, raw


@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.joblib')
    logistic = joblib.load('models/logistic_regression.joblib')
    rf = joblib.load('models/random_forest.joblib')
    xgb = joblib.load('models/xgboost.joblib')
    return scaler, logistic, rf, xgb


X_test, y_test, X_train, raw_df = load_data()
scaler, logistic_model, rf_model, xgb_model = load_models()

X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Churn Analysis")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        options=["Overview", "Model Performance"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<small style='color:#4b5563'>Telco Customer Dataset<br>7,043 customers</small>", unsafe_allow_html=True)


# --------------------------------------------------
# ── PAGE 1: OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.markdown('<div class="page-title">Customer Churn Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Telco customer base · 7,043 accounts</div>', unsafe_allow_html=True)

    # KPI metrics
    total_customers = len(raw_df)
    churn_count = raw_df['Churn'].value_counts().get('Yes', 0)
    churn_rate = round((churn_count / total_customers) * 100, 1)
    retained = total_customers - churn_count
    avg_tenure_churned = round(raw_df[raw_df['Churn'] == 'Yes']['tenure'].mean(), 1)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-sub">full dataset</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churned</div>
            <div class="metric-value warn">{churn_count:,}</div>
            <div class="metric-sub">{churn_rate}% of base</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Retained</div>
            <div class="metric-value">{retained:,}</div>
            <div class="metric-sub">{round(100 - churn_rate, 1)}% of base</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Tenure at Churn</div>
            <div class="metric-value">{avg_tenure_churned}</div>
            <div class="metric-sub">months</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------ Charts row ------
    col_left, col_right = st.columns(2)

    # Chart 1: Churn by contract type
    with col_left:
        st.markdown("## Churn Rate by Contract Type")

        contract_data = raw_df.groupby('Contract')['Churn'].apply(
            lambda col: (col == 'Yes').sum() / len(col) * 100
        )

        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        bars = ax.barh(
            contract_data.index,
            contract_data.values,
            color=['#b45309', '#d97706', '#fbbf24'],
            height=0.5
        )

        ax.set_xlabel('Churn Rate (%)', fontsize=9, color='#6b7280')
        ax.tick_params(colors='#374151', labelsize=9)
        ax.xaxis.label.set_color('#6b7280')
        for spine in ax.spines.values():
            spine.set_color('#e4e0d8')
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax.grid(axis='x', color='#f0ece4', linewidth=0.8)
        ax.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2: Churn by tenure bucket
    with col_right:
        st.markdown("## Churn Rate by Tenure")

        raw_df_copy = raw_df.copy()
        raw_df_copy['tenure_bucket'] = pd.cut(
            raw_df_copy['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0–12 mo', '13–24 mo', '25–48 mo', '49–72 mo'],
            include_lowest=True
        )

        tenure_churn = raw_df_copy.groupby('tenure_bucket', observed=True)['Churn'].apply(
            lambda col: (col == 'Yes').sum() / len(col) * 100
        )

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        fig2.patch.set_facecolor('#ffffff')
        ax2.set_facecolor('#ffffff')

        ax2.bar(
            tenure_churn.index.astype(str),
            tenure_churn.values,
            color='#1d4ed8',
            width=0.5
        )

        ax2.set_ylabel('Churn Rate (%)', fontsize=9, color='#6b7280')
        ax2.tick_params(colors='#374151', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color('#e4e0d8')
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax2.grid(axis='y', color='#f0ece4', linewidth=0.8)
        ax2.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ------ Bottom: churn by internet service ------
    st.markdown("## Churn by Internet Service")

    internet_churn = raw_df.groupby('InternetService')['Churn'].apply(
        lambda col: (col == 'Yes').sum() / len(col) * 100
    ).reset_index()
    internet_churn.columns = ['Internet Service', 'Churn Rate (%)']
    internet_churn['Churn Rate (%)'] = internet_churn['Churn Rate (%)'].round(1)

    count_map = raw_df['InternetService'].value_counts().to_dict()
    internet_churn['Customers'] = internet_churn['Internet Service'].map(count_map)

    st.dataframe(
        internet_churn.sort_values('Churn Rate (%)', ascending=False),
        hide_index=True,
        use_container_width=True
    )


# --------------------------------------------------
# ── PAGE 2: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Logistic Regression · Random Forest · XGBoost — evaluated on 20% holdout set</div>', unsafe_allow_html=True)

    # Get predictions
    lr_proba = logistic_model.predict_proba(X_test_scaled)[:, 1]
    lr_labels = logistic_model.predict(X_test_scaled)

    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_labels = rf_model.predict(X_test_scaled)

    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_labels = xgb_model.predict(X_test_scaled)

    lr_auc = roc_auc_score(y_test, lr_proba)
    rf_auc = roc_auc_score(y_test, rf_proba)
    xgb_auc = roc_auc_score(y_test, xgb_proba)

    # AUC metric cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Logistic Regression</div>
            <div class="metric-value">{round(lr_auc, 4)}</div>
            <div class="metric-sub">AUC-ROC <span class="badge">Best</span></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Random Forest</div>
            <div class="metric-value">{round(rf_auc, 4)}</div>
            <div class="metric-sub">AUC-ROC</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">XGBoost</div>
            <div class="metric-value">{round(xgb_auc, 4)}</div>
            <div class="metric-sub">AUC-ROC</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------ ROC Curve ------
    st.markdown("## ROC Curve")

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    fig3.patch.set_facecolor('#ffffff')
    ax3.set_facecolor('#ffffff')

    ax3.plot(lr_fpr, lr_tpr, color='#1d4ed8', linewidth=2,
             label='Logistic Regression  (AUC = ' + str(round(lr_auc, 4)) + ')')
    ax3.plot(rf_fpr, rf_tpr, color='#b45309', linewidth=2,
             label='Random Forest        (AUC = ' + str(round(rf_auc, 4)) + ')')
    ax3.plot(xgb_fpr, xgb_tpr, color='#059669', linewidth=2,
             label='XGBoost              (AUC = ' + str(round(xgb_auc, 4)) + ')')
    ax3.plot([0, 1], [0, 1], color='#d1c9b8', linewidth=1,
             linestyle='--', label='Random Guess')

    ax3.set_xlabel('False Positive Rate', fontsize=9, color='#6b7280')
    ax3.set_ylabel('True Positive Rate', fontsize=9, color='#6b7280')
    ax3.tick_params(colors='#374151', labelsize=9)
    for spine in ax3.spines.values():
        spine.set_color('#e4e0d8')
    ax3.grid(color='#f0ece4', linewidth=0.8)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=8, frameon=True, facecolor='#ffffff',
               edgecolor='#e4e0d8', loc='lower right')

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ------ Classification reports ------
    st.markdown("## Classification Reports")

    model_choice = st.selectbox(
        "Select model",
        options=["Logistic Regression", "Random Forest", "XGBoost"],
        label_visibility="collapsed"
    )

    if model_choice == "Logistic Regression":
        chosen_labels = lr_labels
    elif model_choice == "Random Forest":
        chosen_labels = rf_labels
    else:
        chosen_labels = xgb_labels

    report_dict = classification_report(
        y_test,
        chosen_labels,
        target_names=['No Churn', 'Churn'],
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.drop(index=['accuracy'], errors='ignore')
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
    report_df['support'] = report_df['support'].astype('Int64')
    report_df = report_df.round(3)

    st.dataframe(report_df, use_container_width=True)

    # ------ Confusion matrix ------
    st.markdown("## Confusion Matrix — " + model_choice)

    cm = confusion_matrix(y_test, chosen_labels)

    cm_df = pd.DataFrame(
        cm,
        index=['Actual: No Churn', 'Actual: Churn'],
        columns=['Predicted: No Churn', 'Predicted: Churn']
    )

    st.dataframe(cm_df, use_container_width=True)
