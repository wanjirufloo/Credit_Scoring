import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("final_xgb_model.pkl")

st.title("African Credit Scoring")
st.write("Predict loan default risk using a machine learning model.")

st.divider()

st.subheader("Loan Information")

loan_type = st.selectbox(
    "Loan Type",
    ["Type_1", "Type_2", "Type_4", "Type_5", "Type_6",
     "Type_7", "Type_9", "Type_10", "Type_11", "Type_12",
     "Type_13", "Type_14", "Type_15", "Type_16", "Type_17",
     "Type_18", "Type_19", "Type_20", "Type_21", "Type_22",
     "Type_23", "Type_24"]
)

total_amount = st.number_input(
    "Total Loan Amount",
    min_value=0.0,
    value=5000.0
)

total_amount_to_repay = st.number_input(
    "Total Amount to Repay",
    min_value=0.0,
    value=5500.0
)

duration = st.number_input(
    "Loan Duration (days)",
    min_value=1,
    value=30
)

new_repeat = st.selectbox(
    "Loan Type",
    ["New Loan", "Repeat Loan"]
)

lender_id = st.selectbox(
    "Lender",
    [245684, 251804, 267277, 267278]
)

amount_funded = st.number_input(
    "Amount Funded By Lender",
    min_value=0.0,
    value=1000.0
)

lender_portion_funded = st.number_input(
    "Lender Portion Funded",
    min_value=0.0,
    max_value=1.0,
    value=0.2
)

lender_portion_repaid = st.number_input(
    "Lender Portion to be Repaid",
    min_value=0.0,
    value=1100.0
)

disbursement_month = st.number_input(
    "Disbursement Month",
    min_value=1,
    max_value=12,
    value=10
)

disbursement_year = st.number_input(
    "Disbursement Year",
    min_value=2021,
    max_value=2025,
    value=2023
)

if st.button("Predict Default Risk"):

    repayment_extra = (
        total_amount_to_repay - total_amount
    )

    if total_amount == 0:
        repayment_ratio = 0
    else:
        repayment_ratio = (
            total_amount_to_repay / total_amount
        )

    input_data = pd.DataFrame({
        "lender_id": [lender_id],
        "loan_type": [loan_type],
        "Total_Amount": [total_amount],
        "Total_Amount_to_Repay": [total_amount_to_repay],
        "duration": [duration],
        "New_versus_Repeat": [new_repeat],
        "Amount_Funded_By_Lender": [amount_funded],
        "Lender_portion_Funded": [lender_portion_funded],
        "Lender_portion_to_be_repaid": [lender_portion_repaid],
        "repayment_extra": [repayment_extra],
        "repayment_ratio": [repayment_ratio],
        "disbursement_month": [disbursement_month],
        "disbursement_year": [disbursement_year]
    })

    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= 0.55)

    st.divider()

    st.subheader("Prediction")

    st.metric(
        "Probability of Default",
        f"{probability:.2%}"
    )

    if prediction == 1:
        st.error("⚠️ HIGHER DEFAULT RISK")
    else:
        st.success("✅ LOWER DEFAULT RISK")

    st.caption(
        "Classification threshold: 0.55"
    )
