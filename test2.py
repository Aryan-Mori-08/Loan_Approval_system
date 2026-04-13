import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ─────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Load pipeline
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

pipeline          = load_pipeline()
num_imp           = pipeline["num_imputer"]
cat_imp           = pipeline["cat_imputer"]
# NOTE: le (LabelEncoder) was re-fit on Loan_Approved in training code,
#       so it cannot encode Education_Level at inference.
#       We use a manual mapping instead (see below).
ohe               = pipeline["onehot_encoder"]
scaler            = pipeline["scaler"]
model             = pipeline["model"]
columns_after_ohe = pipeline["columns_after_ohe"]

# Manual mapping that mirrors what LabelEncoder did during training
# (LabelEncoder sorts labels alphabetically: Graduate=0, Not Graduate=1)
EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a3c6e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a3c6e;
        border-left: 4px solid #1a3c6e;
        padding-left: 10px;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }
    div[data-testid="stButton"] button {
        background-color: #1a3c6e;
        color: white;
        font-size: 1.1rem;
        padding: 0.6rem 2.5rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: background 0.2s;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #2a5ca8;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">🏦 Loan Approval Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Fill in the applicant details below and click <b>Predict</b> to see the result.</p>',
    unsafe_allow_html=True,
)

st.divider()

# ─────────────────────────────────────────────
#  Input form
# ─────────────────────────────────────────────
with st.form("loan_form"):

    # --- Financial Info ---
    st.markdown('<p class="section-title">💰 Financial Information</p>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        Applicant_Income    = st.number_input("Applicant Income (₹)", min_value=0.0, step=1000.0, format="%.2f")
        Loan_Amount         = st.number_input("Loan Amount (₹)",      min_value=0.0, step=1000.0, format="%.2f")

    with fc2:
        Coapplicant_Income  = st.number_input("Coapplicant Income (₹)", min_value=0.0, step=500.0, format="%.2f")
        Loan_Term           = st.number_input("Loan Term (months)",      min_value=0,  step=1)

    with fc3:
        Savings             = st.number_input("Savings Amount (₹)",   min_value=0.0, step=500.0, format="%.2f")
        Collateral_Value    = st.number_input("Collateral Value (₹)", min_value=0.0, step=1000.0, format="%.2f")

    # --- Credit Profile ---
    st.markdown('<p class="section-title">📊 Credit Profile</p>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)

    with cc1:
        Credit_Score    = st.number_input("Credit Score",   min_value=300,  max_value=900, step=1, value=650)
        DTI_Ratio       = st.number_input("DTI Ratio",      min_value=0.0,  max_value=1.0, step=0.01, format="%.2f")

    with cc2:
        Existing_Loans  = st.number_input("Existing Loans", min_value=0, step=1)

    # --- Personal Info ---
    st.markdown('<p class="section-title">👤 Personal Information</p>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        Age             = st.number_input("Age",        min_value=18, max_value=80, step=1, value=30)
        Dependents      = st.number_input("Dependents", min_value=0,  step=1)
        Gender          = st.selectbox("Gender",         ["Male", "Female"])

    with pc2:
        Marital_Status  = st.selectbox("Marital Status",    ["Single", "Married"])
        Education_Level = st.selectbox("Education Level",   ["Graduate", "Not Graduate"])
        Employer_Category = st.selectbox("Employer Category", ["Private", "Government", "Other"])

    with pc3:
        Employment_Status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-employed"])
        Loan_Purpose      = st.selectbox("Loan Purpose",      ["Home", "Car", "Education", "Business"])
        Property_Area     = st.selectbox("Property Area",     ["Urban", "Semiurban", "Rural"])

    st.write("")
    submitted = st.form_submit_button("🔍 Predict Loan Approval")

# ─────────────────────────────────────────────
#  Prediction pipeline
# ─────────────────────────────────────────────
if submitted:

    # ── Step 1: Build raw input (NO Applicant_ID — it was dropped before training) ──
    input_data = pd.DataFrame({
        "Applicant_Income":   [Applicant_Income],
        "Coapplicant_Income": [Coapplicant_Income],
        "Age":                [Age],
        "Dependents":         [Dependents],
        "Existing_Loans":     [Existing_Loans],
        "Savings":            [Savings],
        "Collateral_Value":   [Collateral_Value],
        "Loan_Amount":        [Loan_Amount],
        "Loan_Term":          [Loan_Term],
        "Education_Level":    [Education_Level],
        "Employment_Status":  [Employment_Status],
        "Marital_Status":     [Marital_Status],
        "Loan_Purpose":       [Loan_Purpose],
        "Property_Area":      [Property_Area],
        "Gender":             [Gender],
        "Employer_Category":  [Employer_Category],
        "DTI_Ratio":          [DTI_Ratio],
        "Credit_Score":       [Credit_Score],
    })

    # ── Step 2: Encode Education_Level with manual map ──
    # (LabelEncoder in training was re-fit on Loan_Approved, so we cannot use le.transform here)
    input_data["Education_Level"] = input_data["Education_Level"].map(EDUCATION_MAP)

    # ── Step 3: One-Hot Encode categorical columns ──
    ohe_cols     = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                    "Property_Area", "Gender", "Employer_Category"]
    encoded      = ohe.transform(input_data[ohe_cols])
    encoded_df   = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(ohe_cols),
        index=input_data.index,
    )
    input_data   = pd.concat([input_data.drop(columns=ohe_cols), encoded_df], axis=1)

    # ── Step 4: Feature engineering (MUST match training) ──
    input_data["DTI_ratio_sq"]    = input_data["DTI_Ratio"]    ** 2
    input_data["Credit_Score_sq"] = input_data["Credit_Score"] ** 2

    # ── Step 5: Drop original DTI_Ratio & Credit_Score (dropped in training X) ──
    input_data = input_data.drop(columns=["DTI_Ratio", "Credit_Score"])

    # ── Step 6: Align columns exactly as during training ──
    input_data = input_data.reindex(columns=columns_after_ohe, fill_value=0)

    # ── Step 7: Scale ──
    input_scaled = scaler.transform(input_data)

    # ── Step 8: Predict ──
    prediction   = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence   = max(probabilities) * 100

    # ─────────────────────────────────────────────
    #  Result display
    # ─────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Prediction Result")

    res_col, prob_col = st.columns([1, 1])

    with res_col:
        if prediction == 1:
            st.success("✅  Loan Approved", icon="🎉")
            st.markdown(
                "<div class='result-box' style='background:#d4edda;color:#155724'>"
                "Congratulations! The loan is likely to be approved."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.error("❌  Loan Rejected", icon="🚫")
            st.markdown(
                "<div class='result-box' style='background:#f8d7da;color:#721c24'>"
                "The loan application does not meet the approval criteria."
                "</div>",
                unsafe_allow_html=True,
            )

    with prob_col:
        st.metric("Model Confidence", f"{confidence:.2f}%")
        approved_prob = probabilities[1] * 100
        rejected_prob = probabilities[0] * 100
        prob_df = pd.DataFrame(
            {"Outcome": ["Approved", "Rejected"], "Probability (%)": [approved_prob, rejected_prob]}
        )
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Summary of inputs
    with st.expander("📄 View Submitted Application Details"):
        summary = pd.DataFrame({
            "Field": [
                "Applicant Income", "Coapplicant Income", "Age", "Dependents",
                "Existing Loans", "Savings", "Collateral Value", "Loan Amount",
                "Loan Term", "Credit Score", "DTI Ratio",
                "Education Level", "Employment Status", "Marital Status",
                "Loan Purpose", "Property Area", "Gender", "Employer Category",
            ],
            "Value": [
                f"₹{Applicant_Income:,.2f}", f"₹{Coapplicant_Income:,.2f}",
                Age, Dependents, Existing_Loans,
                f"₹{Savings:,.2f}", f"₹{Collateral_Value:,.2f}",
                f"₹{Loan_Amount:,.2f}", f"{int(Loan_Term)} months",
                Credit_Score, f"{DTI_Ratio:.2f}",
                Education_Level, Employment_Status, Marital_Status,
                Loan_Purpose, Property_Area, Gender, Employer_Category,
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)