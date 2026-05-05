import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(page_title="Churn AI | Batch Predictor", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("📂 Customer Churn Batch Prediction")

# Path handling
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "..", "outputs", "model", "churn_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Sidebar
with st.sidebar:
    st.header("Help & Documentation")
    st.write("Ensure your CSV includes the following columns:")
    st.code("tenure, MonthlyCharges, Contract, PaperlessBilling, PaymentMethod...")
    st.divider()
    st.write("Built for Customer Success Teams")

uploaded_file = st.file_uploader("Upload Customer Data (CSV format)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("1. Data Overview")
    st.dataframe(df.head(5), use_container_width=True)

    # ACTION BUTTON
    if st.button("🚀 Run Analysis"):
        with st.spinner('Analyzing customer behavior...'):
            try:
                # 1. Feature Engineering logic
                df_process = df.copy()
                if 'TotalCharges' not in df_process.columns:
                    df_process['TotalCharges'] = pd.to_numeric(df_process['MonthlyCharges'], errors='coerce') * pd.to_numeric(df_process['tenure'], errors='coerce')
                
                df_process['IsNewCustomer'] = (df_process['tenure'] < 12).astype(int)
                df_process['AvgMonthlySpend'] = df_process['TotalCharges'] / (df_process['tenure'] + 1)
                df_process['HighValueCustomer'] = (df_process['MonthlyCharges'] > 70.0).astype(int)

                # 2. Prediction
                probs = pipeline.predict_proba(df_process)[:, 1]

                # 3. Create ALL Result Columns
                df['Churn_Probability'] = (probs * 100).round(2)
                df['Risk_Level'] = df['Churn_Probability'].apply(
                    lambda x: 'High Risk' if x > 60 else ('Medium Risk' if x > 30 else 'Low Risk')
                )
                df['Display_Risk'] = df['Churn_Probability'].apply(
                    lambda x: '🔴 High Risk' if x > 60 else ('🟡 Medium Risk' if x > 30 else '🟢 Low Risk')
                )

                # 4. Summary Metrics
                st.subheader("2. Insights Summary")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Customers", len(df))
                m2.metric("High Risk Alerts", len(df[df['Risk_Level'] == 'High Risk']))
                m3.metric("Avg. Churn Prob.", f"{df['Churn_Probability'].mean():.1f}%")

                # 5. Detailed Predictions with Styling
                st.subheader("3. Detailed Predictions")
                
                def highlight_risk(row):
                    # This now works because we style the whole DF before hiding columns
                    if row['Risk_Level'] == 'High Risk':
                        return ['background-color: #ffe5e5'] * len(row)
                    return [''] * len(row)

                # Apply styling to the WHOLE dataframe first
                styled_df = df.style.apply(highlight_risk, axis=1)

                # Define the order and which columns to actually SHOW
                # We exclude 'Risk_Level' (plain text) because 'Display_Risk' (emoji) is better for UI
                show_cols = ['Display_Risk', 'Churn_Probability'] + [c for c in df.columns if c not in ['Display_Risk', 'Churn_Probability', 'Risk_Level']]
                
                # Display only the selected columns, but using the styling we generated
                st.dataframe(
                    styled_df, 
                    column_order=show_cols,
                    width="stretch" 
                )

                # 6. Clean Download (Strip UI columns)
                export_df = df.drop(columns=['Display_Risk'])
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Clean Report for Excel",
                    data=csv,
                    file_name='churn_predictions_report.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Critical error during processing: {e}")