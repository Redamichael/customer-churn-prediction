import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title="Churn AI | Interactive Simulator", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px;
                border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn AI – Live Decision Simulator")

# ==========================================
# LOAD MODEL
# ==========================================

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "..", "outputs", "model", "churn_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

pipeline = load_model()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("⚙️ How it works")
    st.write("""
    - Upload customer dataset  
    - Adjust churn threshold live  
    - Run prediction  
    - See business impact instantly  
    """)
    st.divider()
    st.write("📌 Model: RandomForest Pipeline")
    st.write("📌 Purpose: Churn risk simulation tool")

# ==========================================
# UPLOAD DATA
# ==========================================

uploaded_file = st.file_uploader("📂 Upload Customer Data (CSV)", type="csv")

# ==========================================
# LIVE THRESHOLD (IMPORTANT FIX)
# ==========================================

threshold = st.slider(
    "🎛️ Churn Decision Threshold (Live Control)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

# ==========================================
# RUN ONLY IF FILE EXISTS
# ==========================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("1. Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("🚀 Run Analysis"):

        with st.spinner("Running churn prediction..."):

            df_process = df.copy()

            # ==========================================
            # FEATURE ENGINEERING (MATCH TRAINING)
            # ==========================================

            if 'TotalCharges' not in df_process.columns:
                df_process['TotalCharges'] = (
                    pd.to_numeric(df_process['MonthlyCharges'], errors='coerce') *
                    pd.to_numeric(df_process['tenure'], errors='coerce')
                )

            df_process['IsNewCustomer'] = (df_process['tenure'] < 12).astype(int)
            df_process['AvgMonthlySpend'] = df_process['TotalCharges'] / (df_process['tenure'] + 1)
            df_process['HighValueCustomer'] = (df_process['MonthlyCharges'] > 70).astype(int)

            # ==========================================
            # PREDICTION
            # ==========================================

            probs = pipeline.predict_proba(df_process)[:, 1]
            df['Churn_Probability'] = probs

            # ==========================================
            # THRESHOLD APPLICATION (LIVE)
            # ==========================================

            df['Predicted_Churn'] = (df['Churn_Probability'] >= threshold).astype(int)

            df['Risk_Level'] = df['Predicted_Churn'].apply(
                lambda x: "High Risk" if x == 1 else "Low Risk"
            )

            df['Display_Risk'] = df['Predicted_Churn'].apply(
                lambda x: "🔴 High Risk" if x == 1 else "🟢 Low Risk"
            )

            # ==========================================
            # METRICS
            # ==========================================

            st.subheader("📊 Live Business Impact")

            flagged_pct = df['Predicted_Churn'].mean() * 100
            avg_prob = df['Churn_Probability'].mean() * 100

            c1, c2, c3 = st.columns(3)

            c1.metric("Flagged Customers (%)", f"{flagged_pct:.1f}%")
            c2.metric("Avg Churn Probability", f"{avg_prob:.1f}%")
            c3.metric("Threshold", f"{threshold:.2f}")

            # ==========================================
            # PRECISION–RECALL CURVE
            # ==========================================

            st.subheader("📉 Precision–Recall Tradeoff")

            precision, recall, thresholds = precision_recall_curve(
                df['Predicted_Churn'],
                df['Churn_Probability']
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=thresholds,
                y=precision[:-1],
                name="Precision",
                line=dict(width=3)
            ))

            fig.add_trace(go.Scatter(
                x=thresholds,
                y=recall[:-1],
                name="Recall",
                line=dict(width=3)
            ))

            fig.add_vline(
                x=threshold,
                line_width=2,
                line_dash="dash",
                line_color="black"
            )

            fig.update_layout(
                height=400,
                xaxis_title="Threshold",
                yaxis_title="Score",
                title="Precision vs Recall Tradeoff"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ==========================================
            # CUSTOMER TABLE
            # ==========================================

            st.subheader("👥 Customer Risk Table")

            def highlight(row):
                return [
                    'background-color: #ffe5e5' if row['Predicted_Churn'] == 1 else ''
                    for _ in row
                ]

            show_cols = ['Display_Risk'] + [c for c in df.columns if c != 'Display_Risk']
            styled_df = df[show_cols].style.apply(highlight, axis=1)

            st.dataframe(styled_df, use_container_width=True)

            # ==========================================
            # DOWNLOAD
            # ==========================================

            export_df = df.drop(columns=['Display_Risk'])
            csv = export_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Report",
                data=csv,
                file_name='churn_predictions_live.csv',
                mime='text/csv'
            )