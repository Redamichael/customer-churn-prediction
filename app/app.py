# customer-churn-prediction
import streamlit as st
import pandas as pd 
import joblib 
st.set_page_config(page_title="Churn Dashboard", layout="wide") 
st.title("Customer Churn Prediction") 
pipeline = joblib.load("../outputs/model/churn_pipeline.pkl") 
st.sidebar.header("Customer Input") 
tenure = st.sidebar.slider("Tenure", 0, 72, 12) 
monthly = st.sidebar.slider("Monthly Charges", 0, 150, 50) 
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", 
"Two year"]) 
input_df = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["No"],
        "Dependents": ["No"],
        "tenure": [tenure],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": [contract],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [monthly],
        "TotalCharges": [monthly * tenure]
    })
input_df['IsNewCustomer'] = (input_df['tenure'] < 12).astype(int)
input_df['AvgMonthlySpend'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)
median_threshold = 70.0
input_df['HighValueCustomer'] = (input_df['MonthlyCharges'] > median_threshold).astype(int)
if st.sidebar.button("Predict"): 
    prob = pipeline.predict_proba(input_df)[0][1] 
    st.metric("Churn Probability", f"{prob:.2%}") 
    if prob > 0.6: 
        st.error("High Risk") 
    elif prob > 0.3: 
        st.warning("Medium Risk") 
    else: 
     st.success("Low Risk")