import os
import streamlit as st
import requests
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | TelcoML",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Main background */
    .stApp {
        background: #0d1117;
        color: #c9d1d9;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        text-align: center;
    }
    .hero-banner h1 {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-banner p {
        color: #8b949e;
        font-size: 1.05rem;
    }

    /* Section headers */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #8b949e;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin: 20px 0 16px 0;
    }

    /* Form card */
    .form-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }

    /* Input labels */
    label { color: #c9d1d9 !important; font-size: 0.88rem !important; }

    /* Predict button */
    .stFormSubmitButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 14px 0;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-top: 20px;
        transition: opacity 0.2s;
    }
    .stFormSubmitButton > button:hover { opacity: 0.88; }

    /* Result card */
    .result-churn {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #dc2626;
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #16a34a;
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-label {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-sub {
        font-size: 1rem;
        color: #d1d5db;
    }

    /* Number inputs & select boxes */
    .stNumberInput input, .stSelectbox select {
        background: #0d1117 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* Hide Streamlit default header */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── FastAPI endpoint ───────────────────────────────────────────────────────────
# In production: set BACKEND_URL env var on Streamlit Cloud to your Render URL
# e.g. https://customer-churn-api.onrender.com
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000") + "/predict"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 TelcoML")
    st.markdown("---")
    st.markdown("""
    **About this app**  
    This tool uses an ensemble of **Logistic Regression**, **Random Forest**, and **XGBoost** models 
    to predict the probability of a telecom customer churning.
    
    ---
    **Model Metrics**
    - 🎯 ROC-AUC: **0.8399**
    - ✅ Accuracy: **72.99%**
    - 📦 Type: **Soft Voting Ensemble**
    
    ---
    **Stack**
    - FastAPI (Backend)
    - Streamlit (Frontend)
    - MLflow + DagsHub (Tracking)
    - DVC (Versioning)
    """)

# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>📡 Customer Churn Predictor</h1>
    <p>Powered by an MLOps-grade ensemble model — fill in customer details and get an instant churn risk assessment.</p>
</div>
""", unsafe_allow_html=True)

# ── Prediction Form ────────────────────────────────────────────────────────────
with st.form("churn_form"):

    # --- Demographics ---
    st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with d2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with d3:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
    with d4:
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    # --- Account & Billing ---
    st.markdown('<div class="section-title">💳 Account & Billing</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        tenure = st.number_input("Tenure (Months)", min_value=0.0, max_value=100.0, value=12.0, step=1.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with a2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=75.5)
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with a3:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=25000.0, value=900.0)
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    # --- Services ---
    st.markdown('<div class="section-title">🌐 Services</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with s2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with s3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("🔮  Predict Churn Risk")

# ── Prediction Logic ───────────────────────────────────────────────────────────
if submitted:
    def normalize(val):
        return "No" if "No" in val else "Yes"

    # Strip "(automatic)" from payment method to match schema
    clean_payment = payment_method.split(" (")[0]

    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": normalize(multiple_lines),
        "InternetService": internet_service,
        "OnlineSecurity": normalize(online_security),
        "OnlineBackup": normalize(online_backup),
        "DeviceProtection": normalize(device_protection),
        "TechSupport": normalize(tech_support),
        "StreamingTV": normalize(streaming_tv),
        "StreamingMovies": normalize(streaming_movies),
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": clean_payment
    }

    with st.spinner("🔍 Analyzing customer profile..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                prob = result["probability"]
                confidence = result["confidence"]
                label = result["label"]
                prediction = result["prediction"]

                st.markdown("---")
                r_col, g_col = st.columns([1, 1])

                # ── Result Card ──
                with r_col:
                    if label == 1:
                        st.markdown(f"""
                        <div class="result-churn">
                            <div class="result-label">⚠️ High Churn Risk</div>
                            <div class="result-sub">{prediction}<br><br>
                            Confidence: <strong>{confidence*100:.1f}%</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-safe">
                            <div class="result-label">✅ Low Churn Risk</div>
                            <div class="result-sub">{prediction}<br><br>
                            Confidence: <strong>{confidence*100:.1f}%</strong></div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Gauge Chart ──
                with g_col:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(prob * 100, 1),
                        number={"suffix": "%", "font": {"size": 40, "color": "white"}},
                        title={"text": "Churn Probability", "font": {"size": 18, "color": "#8b949e"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
                            "bar": {"color": "#dc2626" if label == 1 else "#16a34a", "thickness": 0.3},
                            "bgcolor": "#161b22",
                            "bordercolor": "#30363d",
                            "steps": [
                                {"range": [0, 30], "color": "#052e16"},
                                {"range": [30, 60], "color": "#422006"},
                                {"range": [60, 100], "color": "#450a0a"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "thickness": 0.8,
                                "value": 50
                            }
                        }
                    ))
                    fig.update_layout(
                        paper_bgcolor="#0d1117",
                        plot_bgcolor="#0d1117",
                        font={"color": "white"},
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ── Key factors callout ──
                st.markdown("---")
                st.markdown("#### 💡 Key Risk Factors in this Profile")
                tips = []
                if contract == "Month-to-month":
                    tips.append("📋 **Month-to-month contract** — highest churn-risk contract type (~43% churn rate)")
                if internet_service == "Fiber optic":
                    tips.append("📶 **Fiber optic internet** — associated with ~42% churn rate")
                if clean_payment == "Electronic check":
                    tips.append("💳 **Electronic check payment** — highest churn payment method (~45%)")
                if tech_support == "No" and online_security == "No":
                    tips.append("🛡️ **No TechSupport / OnlineSecurity** — lack of add-ons triples churn risk")
                if tenure < 12:
                    tips.append(f"📅 **Short tenure ({int(tenure)} months)** — new customers are at highest risk")
                if tips:
                    for tip in tips:
                        st.warning(tip)
                else:
                    st.success("✅ No major risk factors detected in this profile.")

            else:
                st.error(f"❌ Prediction service error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ **Cannot reach the FastAPI backend** at `http://127.0.0.1:8000`.\n\n"
                "Make sure you start it first:\n```bash\nuvicorn main:app --reload\n```"
            )
        except requests.exceptions.Timeout:
            st.error("⏱️ The prediction request timed out. Please try again.")
