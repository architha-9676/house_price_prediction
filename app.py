# ============================================================
#   app.py - Streamlit Web Dashboard (FIXED)
#   House Price Prediction | QSkill Internship
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .predict-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    .predict-price {
        font-size: 3rem;
        font-weight: 900;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f4e79, #2980b9);
        color: white;
        border: none;
        padding: 0.7rem 2.5rem;
        font-size: 1.1rem;
        border-radius: 8px;
        width: 100%;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model        = joblib.load('models/house_price_model.pkl')
        scaler       = joblib.load('models/scaler.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        return model, scaler, feature_cols
    except Exception as e:
        return None, None, None


# ============================================================
#   MAIN APP
# ============================================================

st.markdown('<div class="main-title">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">QSkill Internship | Python Development | ML Model</div>', unsafe_allow_html=True)
st.markdown("---")

model, scaler, feature_cols = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run `python model.py` first.")
    st.code("python model.py", language="bash")
    st.stop()

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Data Analysis", "📈 Model Performance"])


# ============================================================
#   TAB 1: PREDICT
# ============================================================
with tab1:
    st.subheader("Enter House Details to Predict Price")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms    = st.slider("🛏️ Bedrooms",   1, 8, 3)
        bathrooms   = st.slider("🚿 Bathrooms",   1, 5, 2)
        floors      = st.select_slider("🏢 Floors", [1, 1.5, 2, 2.5, 3], value=2)
        waterfront  = st.selectbox("🌊 Waterfront", ["No", "Yes"])

    with col2:
        sqft_living = st.number_input("📐 Living Area (sqft)", 300, 10000, 2000, step=100)
        sqft_lot    = st.number_input("🌿 Lot Size (sqft)",    500, 50000, 5000, step=500)
        sqft_above  = st.number_input("🏗️ Sqft Above Ground",  300, 10000, 1500, step=100)
        sqft_basement = st.number_input("🏚️ Sqft Basement",     0,   5000,  500, step=100)

    with col3:
        age         = st.slider("📅 House Age (years)", 0, 100, 10)
        condition   = st.slider("⭐ Condition (1-5)",   1, 5,   3)
        grade       = st.slider("🏅 Grade (1-13)",      1, 13,  7)
        view        = st.slider("👁️ View (0-4)",         0, 4,   0)
        garage      = st.selectbox("🚗 Garage", ["Yes", "No"])

    # Neighbor estimates
    st.markdown("##### 🏘️ Neighborhood (optional estimates)")
    nc1, nc2 = st.columns(2)
    with nc1:
        sqft_living15 = st.number_input("Neighbor Avg Living Sqft", 300, 10000, 1800, step=100)
    with nc2:
        sqft_lot15    = st.number_input("Neighbor Avg Lot Sqft",    500, 50000, 5000, step=500)

    st.markdown("")
    predict_btn = st.button("🔮 Predict House Price")

    if predict_btn:
        wf  = 1 if waterfront == "Yes" else 0
        gar = 1 if garage     == "Yes" else 0

        # Build all possible features
        raw = {
            'bedrooms':             bedrooms,
            'bathrooms':            bathrooms,
            'sqft_living':          sqft_living,
            'sqft_lot':             sqft_lot,
            'floors':               floors,
            'waterfront':           wf,
            'view':                 view,
            'condition':            condition,
            'grade':                grade,
            'sqft_above':           sqft_above,
            'sqft_basement':        sqft_basement,
            'sqft_living15':        sqft_living15,
            'sqft_lot15':           sqft_lot15,
            'age':                  age,
            'was_renovated':        0,
            'garage':               gar,
            'living_lot_ratio':     sqft_living / (sqft_lot + 1),
            'living_vs_neighbors':  sqft_living / (sqft_living15 + 1),
            'lot_vs_neighbors':     sqft_lot    / (sqft_lot15 + 1),
            'sqft_per_bath':        sqft_living / (bathrooms + 1),
            'bed_bath_ratio':       bedrooms    / (bathrooms + 1),
            'sqft_per_floor':       sqft_living / (floors + 1),
            'grade_sqft':           grade       * sqft_living,
            'view_waterfront':      view        * (wf + 1),
        }

        # Only keep features model was trained on
        input_data = {k: raw[k] for k in feature_cols if k in raw}
        input_df   = pd.DataFrame([input_data])[feature_cols]

        input_scaled = scaler.transform(input_df)
        predicted    = model.predict(input_scaled)[0]

        # If model was trained on log price, convert back
        if predicted < 100:
            predicted = np.expm1(predicted)

        st.markdown(f"""
        <div class="predict-box">
            <div style="font-size:1.1rem; margin-bottom:0.5rem;">💰 Estimated House Price</div>
            <div class="predict-price">${predicted:,.0f}</div>
            <div style="font-size:0.9rem; margin-top:0.5rem; opacity:0.85;">
                {bedrooms} bed · {bathrooms} bath · {sqft_living} sqft · Grade {grade}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.metric("📉 Lower Estimate", f"${predicted * 0.90:,.0f}")
        c2.metric("🎯 Predicted Price", f"${predicted:,.0f}")
        c3.metric("📈 Upper Estimate", f"${predicted * 1.10:,.0f}")


# ============================================================
#   TAB 2: DATA ANALYSIS
# ============================================================
with tab2:
    st.subheader("📊 Exploratory Data Analysis")
    if os.path.exists('plots/eda_dashboard.png'):
        st.image('plots/eda_dashboard.png', use_column_width=True)
    else:
        st.info("Run `python model.py` to generate plots.")

    st.markdown("### 🔍 Key Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ **Living Area** has the strongest positive impact on price")
        st.success("✅ **Grade** is the most important feature overall")
        st.success("✅ **Waterfront** properties command a massive premium")
    with col2:
        st.warning("⚠️ **House Age** negatively affects price")
        st.success("✅ **View rating** directly correlates with price")
        st.success("✅ **Condition** and **Grade** together define quality")


# ============================================================
#   TAB 3: MODEL PERFORMANCE
# ============================================================
with tab3:
    st.subheader("📈 Model Performance")
    if os.path.exists('plots/model_performance.png'):
        st.image('plots/model_performance.png', use_column_width=True)
    else:
        st.info("Run `python model.py` to generate plots.")

    st.markdown("### 📋 About the Model")
    st.markdown("""
    | Property | Details |
    |---|---|
    | **Algorithm** | Gradient Boosting (Best of 4 models) |
    | **Models Compared** | Linear Regression, Random Forest, Gradient Boosting, XGBoost |
    | **Features Used** | 24 engineered features |
    | **Dataset** | Real Kaggle KC House Data (21,000+ houses) |
    | **Train/Test Split** | 80% / 20% |
    | **R² Accuracy** | ~74% on real-world data |
    """)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>🏠 House Price Predictor · QSkill Internship · Built with Python & Streamlit</small></center>",
    unsafe_allow_html=True
)
