# African Credit Scoring Challenge

A machine learning project for predicting loan default risk using the African Credit Scoring Challenge dataset from Zindi.

## Project Overview

The goal of this project is to predict whether a loan will default (`target = 1`) or be successfully repaid (`target = 0`).

The dataset is highly imbalanced, with loan defaults representing only approximately 1.83% of the training observations. The project therefore focuses on appropriate evaluation metrics and threshold optimization rather than relying on accuracy alone.

## Approach

- Exploratory Data Analysis (EDA)
- Data quality and structural investigation
- Feature engineering
- Categorical and numerical feature preprocessing
- Stratified train-validation splitting
- Logistic Regression baseline
- Random Forest
- XGBoost
- Class imbalance handling
- F1-score optimization
- Classification threshold optimization
- Cross-validation
- Final prediction generation

## Results

The models achieved the following validation performance:

| Model | F1 Score |
|---|---:|
| Logistic Regression | 0.685 |
| Random Forest | 0.850 |
| XGBoost | 0.851 |

XGBoost achieved a mean cross-validation F1-score of approximately **0.845 ± 0.007** across five stratified folds.

The final XGBoost model used a validation-optimized classification threshold of **0.55**.

## Key Findings

The most influential features in the XGBoost model included:

- Repayment ratio
- Loan type
- Repayment extra
- Disbursement year
- Loan duration
- Amount funded by lender

## Project Structure

```text
credit-scoring/
│
├── data/
├── notebooks/
├── src/
├── outputs/
├── requirements.txt
├── README.md
└── submission.csv
