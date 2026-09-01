import streamlit as st
import pandas as pd
import joblib

# Load the saved model and preprocessing objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

LABELS = {0: "Poor", 1: "Standard", 2: "Good"}

st.title("Credit Score Classifier")

month = st.selectbox("Month", list(encoders["Month"].classes_))
age = st.number_input("Age", min_value=14, max_value=100, value=30)
occupation = st.selectbox("Occupation", list(encoders["Occupation"].classes_))
annual_income = st.number_input("Annual Income", value=50000.0)
monthly_salary = st.number_input("Monthly In-hand Salary", value=4000.0)
bank_accounts = st.number_input("Number of Bank Accounts", value=3)
credit_cards = st.number_input("Number of Credit Cards", value=4)
interest_rate = st.number_input("Interest Rate (%)", value=10)
num_loans = st.number_input("Number of Loans", value=2)
delay_from_due = st.number_input("Delay from Due Date (days)", value=10)
num_delayed_payments = st.number_input("Number of Delayed Payments", value=5.0)
changed_credit_limit = st.number_input("Changed Credit Limit", value=5.0)
credit_inquiries = st.number_input("Number of Credit Inquiries", value=4.0)
credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
outstanding_debt = st.number_input("Outstanding Debt", value=1000.0)
credit_util_ratio = st.number_input("Credit Utilization Ratio", value=30.0)
payment_min_amt = st.selectbox("Payment of Min Amount", ["Yes", "No"])
total_emi = st.number_input("Total EMI per Month", value=100.0)
amount_invested = st.number_input("Amount Invested Monthly", value=100.0)
payment_behaviour = st.selectbox("Payment Behaviour", list(encoders["Payment_Behaviour"].classes_))
monthly_balance = st.number_input("Monthly Balance", value=300.0)

if st.button("Predict Credit Score"):
    row = {
        "Month": encoders["Month"].transform([month])[0],
        "Age": age,
        "Occupation": encoders["Occupation"].transform([occupation])[0],
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": bank_accounts,
        "Num_Credit_Card": credit_cards,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_loans,
        "Delay_from_due_date": delay_from_due,
        "Num_of_Delayed_Payment": num_delayed_payments,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": credit_inquiries,
        "Credit_Mix": {"Good": 2, "Standard": 1, "Bad": 0}[credit_mix],
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_util_ratio,
        "Payment_of_Min_Amount": {"Yes": 1, "No": 0}[payment_min_amt],
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": amount_invested,
        "Payment_Behaviour": encoders["Payment_Behaviour"].transform([payment_behaviour])[0],
        "Monthly_Balance": monthly_balance,
    }
    input_df = pd.DataFrame([row])[feature_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.subheader(f"Predicted Credit Score: {LABELS[prediction]}")