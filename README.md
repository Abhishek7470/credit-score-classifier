# 💳 Credit Score Classifier

A machine learning web app that predicts whether a customer's credit score is **Good**, **Standard**, or **Poor** based on their financial and credit history data.

🔗 **Live Demo:** [credit-score-classifier.streamlit.app](https://credit-score-classifier-8tnjbvkaneuwajfcorj5jg.streamlit.app/)

## Overview

This project trains a classification model on ~100,000 customer records covering income, loans, payment behavior, credit utilization, and more — then serves real-time predictions through an interactive Streamlit web app.

## Tech Stack

- **Python**
- **Pandas / NumPy** — data cleaning and preprocessing
- **Scikit-learn** — Random Forest classifier, label encoding, feature scaling
- **Streamlit** — web app / deployment
- **Joblib** — model serialization

## Model

- Algorithm: Random Forest Classifier
- Preprocessing: missing value imputation, outlier removal (IQR method on Age), label encoding for categorical features, StandardScaler for numeric features
- Accuracy: ~78% on held-out test data

## Project Structure

```
├── app.py                 # Streamlit web app
├── train.py                # Training script (data cleaning + model training)
├── model.pkl                # Trained Random Forest model
├── scaler.pkl                # Fitted StandardScaler
├── encoders.pkl               # Fitted LabelEncoders (Month, Occupation, Payment_Behaviour)
├── feature_columns.pkl         # Feature column order expected by the model
└── requirements.txt            # Python dependencies
```

## Run Locally

```bash
git clone https://github.com/Abhishek7470/credit-score-classifier.git
cd credit-score-classifier
pip install -r requirements.txt
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## How It Works

1. User fills in customer details (income, loans, payment history, etc.) through the form
2. Inputs are encoded and scaled using the same preprocessing pipeline used during training
3. The trained Random Forest model predicts the credit score category and returns class probabilities

## Author

**Abhishekh Bamniya**
[LinkedIn](https://linkedin.com/in/abhishekh-bamniya-973483297) · [GitHub](https://github.com/Abhishekh7470) · [Kaggle](https://kaggle.com/abhishekhbamniya)
