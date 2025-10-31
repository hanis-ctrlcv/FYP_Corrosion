# ============================================================
# Streamlit App: Corrosion Rate Prediction & Pipeline Simulation
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import datetime, os, base64

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="Pipeline Corrosion Status Dashboard", layout="wide")

# ============================================================
# HEADER WITH UTP LOGO
# ============================================================
logo_path = "utp logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{logo_base64}' width='250'>
        <h2 style='margin-top: 10px; color:#004d80;'>Deep Learning Corrosion Prediction Dashboard</h2>
        <p style='color:#666;font-size:16px;'><b>Final Year Project by Muhammad Hanis Afifi Bin Azmi</b></p>
        <hr style='margin-top:10px;margin-bottom:30px;'>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_PATH = "final_corrosion_model.keras"  # ✅ updated
PREPROCESSOR_PATH = "preprocessor_corrosion.joblib"
RF_PATH = "rf_model.joblib"
XGB_PATH = "xgb_model.json"
DATA_PATH = "cleaned_corrosion_regression_data.csv"

st.title("🛠️ Corrosion Monitoring Dashboard")
st.caption("Powered by Reinforced Deep Learning (DL + RF + XGB Ensemble) ✅")

# ============================================================
# LOAD MODELS & DATA SAFELY
# ============================================================
st.sidebar.info(f"TensorFlow version: {tf.__version__}")

try:
    model_dl = tf.keras.models.load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    rf = joblib.load(RF_PATH)
    xgb = XGBRegressor()
    xgb.load_model(XGB_PATH)
    df = pd.read_csv(DATA_PATH)
    st.success("✅ All models and dataset loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or data: {e}")
    st.stop()

# ============================================================
# PIPELINE FIXED DATA
# ============================================================
PIPE_DATA = {
    "PIPE A": pd.DataFrame([{
        "Environment": "Acetic Acid", "Material Family": "Lead and alloys",
        "Concentration_%": 2, "Temperature_C": 20,
        "Pred_DL(mm/yr)": 1.3484257,
        "Pred_RF(mm/yr)": 0.852185268,
        "Pred_XGB(mm/yr)": 0.9962389,
        "Pred_Reinforced Deep Learning(mm/yr)": 1.065616634
    }]),
    "PIPE B": pd.DataFrame([{
        "Environment": "Acetaldehyde", "Material Family": "Titanium and alloys",
        "Concentration_%": 75, "Temperature_C": 149,
        "Pred_DL(mm/yr)": -0.010204196,
        "Pred_RF(mm/yr)": 0.147873478,
        "Pred_XGB(mm/yr)": 0.20677292,
        "Pred_Reinforced Deep Learning(mm/yr)": 0.114814068
    }]),
    "PIPE C": pd.DataFrame([{
        "Environment": "Acetic Acid", "Material Family": "Copper",
        "Concentration_%": 90, "Temperature_C": 40,
        "Pred_DL(mm/yr)": 0.04234609,
        "Pred_RF(mm/yr)": 0.16278569,
        "Pred_XGB(mm/yr)": 0.20677292,
        "Pred_Reinforced Deep Learning(mm/yr)": 0.137301568
    }])
}

# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================
st.subheader("📂 Trained Dataset Overview")
st.dataframe(df.head(), use_container_width=True)

# ============================================================
# SECTION 2 — PIPELINE SIMULATION
# ============================================================
st.subheader("🚧 Pipeline Corrosion Status")

if "pipe_selected" not in st.session_state:
    st.session_state.pipe_selected = "PIPE A"

def pipe_color(pipe_key):
    val = PIPE_DATA[pipe_key]["Pred_Reinforced Deep Learning(mm/yr)"].values[0]
    if val <= 0.1: return "#2ECC71"
    elif val <= 1.0: return "#F1C40F"
    return "#E74C3C"

def render_pipe(pipe_name):
    color = pipe_color(pipe_name)
    selected = (pipe_name == st.session_state.pipe_selected)
    border = "6px solid black" if selected else "3px solid #333"
    return f"""
    <div style="width:130px;height:300px;background:{color};
    border-radius:18px;border:{border};font-weight:bold;color:white;
    font-size:22px;margin:auto;display:flex;align-items:center;
    justify-content:center;cursor:pointer;box-shadow:0 10px 20px rgba(0,0,0,0.4);">
        {pipe_name}
    </div>
    """

col1, col2, col3 = st.columns(3)
for i, pipe in enumerate(["PIPE A", "PIPE B", "PIPE C"]):
    with [col1, col2, col3][i]:
        if st.button(pipe, key=f"pipe_{pipe}"):
            st.session_state.pipe_selected = pipe
        st.markdown(render_pipe(pipe), unsafe_allow_html=True)

pipe_selected = st.session_state.pipe_selected
st.success(f"✅ Selected pipeline: {pipe_selected}")

# ============================================================
# SECTION 3 — PIPE INFO DISPLAY
# ============================================================
pipe_df = PIPE_DATA[pipe_selected]
st.subheader("📋 Predicted Corrosion Rate Summary")
st.dataframe(pipe_df.style.format(precision=4), use_container_width=True)

# ============================================================
# SECTION 4 — CSV UPLOAD FOR BULK PREDICTION
# ============================================================
st.subheader("📥 Upload CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.dataframe(user_df.head())

    if st.button("🔮 Run Predictions"):
        try:
            expected = list(preprocessor.feature_names_in_)
            user_df = user_df.reindex(columns=expected)
            X_new = preprocessor.transform(user_df)

            p_dl = model_dl.predict(X_new).ravel()
            p_rf = rf.predict(X_new)
            p_xgb = xgb.predict(X_new)
            p_ens = (p_dl + p_rf + p_xgb) / 3

            user_df["Pred_Reinforced_DL(mm/yr)"] = p_ens
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("predictions", exist_ok=True)
            out_path = f"predictions/prediction_{timestamp}.csv"
            user_df.to_csv(out_path, index=False)

            st.success(f"✅ Prediction Complete — Saved to {out_path}")
            st.dataframe(user_df.head(10))
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ============================================================
# SECTION 5 — MODEL VISUALIZATION
# ============================================================
st.subheader("📊 Model Prediction Comparison")

X = df.drop(columns=["Rate (mm/yr)"])
y = df["Rate (mm/yr)"]
X_prepared = preprocessor.transform(X)

y_dl = model_dl.predict(X_prepared).ravel()
y_rf = rf.predict(X_prepared)
y_xgb = xgb.predict(X_prepared)
y_ens = (y_dl + y_rf + y_xgb) / 3

df_viz = pd.DataFrame({
    "Actual": y,
    "Deep Learning": y_dl,
    "Random Forest": y_rf,
    "XGBoost": y_xgb,
    "Reinforced Deep Learning": y_ens
})

melted = df_viz.melt(id_vars="Actual", var_name="Model", value_name="Predicted")
fig = px.scatter(melted, x="Actual", y="Predicted", color="Model",
                 title="Actual vs Predicted Corrosion Rate")
fig.add_shape(type="line", x0=y.min(), y0=y.min(),
              x1=y.max(), y1=y.max(),
              line=dict(color="black", dash="dash"))
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SECTION 6 — ACCURACY SUMMARY
# ============================================================
st.subheader("🧠 Reinforced Deep Learning Model Accuracy Summary")

r2_val = r2_score(y, y_ens)
mae_val = mean_absolute_error(y, y_ens)
rmse_val = np.sqrt(mean_squared_error(y, y_ens))
accuracy_pct = r2_val * 100

st.markdown(f"""
<div style='background-color:#E8F6EF;padding:15px;border-radius:10px;margin-top:10px;'>
    <h4 style='text-align:center;color:#1E8449;'>
        ✅ <b>Reinforced Deep Learning Accuracy: {accuracy_pct:.2f}%</b><br>
        (R² = {r2_val:.4f}, MAE = {mae_val:.4f}, RMSE = {rmse_val:.4f})
    </h4>
</div>
""", unsafe_allow_html=True)
# ============================================================
# SECTION X — Remaining Life Estimation
# ============================================================
st.subheader("⏳ Estimated Remaining Life (Based on Predicted Corrosion Rate)")

# Parameters (you can make them adjustable via sliders)
T_CURRENT = 10.0  # mm
T_MIN = 5.0       # mm
MAE = mae_val        # mm/yr
PITTING_FACTOR = 1.5
MC_SAMPLES = 5000
np.random.seed(42)

pipe_pred = PIPE_DATA[pipe_selected].iloc[0]["Pred_Reinforced Deep Learning(mm/yr)"]

# --- Deterministic conservative calculation ---
def deterministic_remaining_life(curr_t, min_t, pred_r, mae=MAE, pitting_factor=PITTING_FACTOR):
    pred_r = max(pred_r, 0.0)
    r_upper = pred_r + mae
    r_eff = r_upper * pitting_factor
    if r_eff <= 1e-12:
        return np.inf, r_upper, r_eff
    remaining = (curr_t - min_t) / r_eff
    if remaining < 0:
        remaining = 0.0
    return remaining, r_upper, r_eff

det_years, r_upper, r_eff = deterministic_remaining_life(T_CURRENT, T_MIN, pipe_pred)

# --- Monte Carlo simulation (uncertainty-aware) ---
def mc_remaining_life(curr_t, min_t, pred_r, sigma=MAE, pf_range=(1.0, 2.0), n_samples=MC_SAMPLES):
    r_samps = np.random.normal(pred_r, sigma, size=n_samples)
    r_samps = np.clip(r_samps, 0.0, None)
    pf_samps = np.random.uniform(pf_range[0], pf_range[1], size=n_samples)
    r_eff = r_samps * pf_samps
    r_eff = np.where(r_eff <= 1e-12, 1e-12, r_eff)
    rem_samps = (curr_t - min_t) / r_eff
    rem_samps = np.clip(rem_samps, 0.0, None)
    return {
        "median": np.median(rem_samps),
        "p05": np.percentile(rem_samps, 5),
        "p95": np.percentile(rem_samps, 95)
    }

mc = mc_remaining_life(T_CURRENT, T_MIN, pipe_pred)

# --- Display results ---
html_block = f"""
<div style='background-color:#f7f9f9;border-radius:12px;padding:15px;
box-shadow:0 4px 10px rgba(0,0,0,0.1);'>
    <h4 style='text-align:center;color:#1E8449;'>
        📏 Remaining Life Estimation for <b>{pipe_selected}</b>
    </h4>
    <p style='text-align:center;font-size:16px;'>
        Based on predicted corrosion rate from Reinforced Deep Learning model
    </p>
    <table style='margin:auto;font-size:16px;border-collapse:collapse;'>
        <tr><th style='text-align:left;padding:8px;'>Predicted Rate (mm/yr)</th>
            <td style='padding:8px;'>{pipe_pred:.4f}</td></tr>
        <tr><th style='text-align:left;padding:8px;'>Effective Rate (with MAE + Pitting)</th>
            <td style='padding:8px;'>{r_eff:.4f}</td></tr>
        <tr><th style='text-align:left;padding:8px;'>Deterministic Remaining Life (yrs)</th>
            <td style='padding:8px;color:#2874A6;font-weight:bold;'>{det_years:.2f}</td></tr>
    </table>
</div>
"""

st.markdown(html_block, unsafe_allow_html=True)
# ============================================================
# SECTION 7 — CORRELATION HEATMAP
# ============================================================
st.subheader("📈 Correlation Heatmap of Features")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ============================================================
# SECTION 8 — PAIRPLOT VISUALIZATION
# ============================================================
st.subheader("🔍 Feature Interaction Overview (Pairplot)")
selected_cols = ["Rate (mm/yr)", "Concentration_%", "Temperature_C", "Aggressiveness_Index"]
sns.pairplot(df[selected_cols], diag_kind="kde", corner=True)
st.pyplot(plt)


