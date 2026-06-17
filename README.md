# Energy Customer Churn - PowerCo Dataset

Churn prediction pipeline for an energy utility â€” LightGBM model with SMOTE balancing and SHAP explainability across client, usage, and pricing data.

## Overview

This notebook predicts which customers are likely to churn at PowerCo, a fictional energy utility. Starting from raw client, price history, and churn label data, it walks through exploratory analysis, feature engineering, model training, and explainability â€” ending with a scored CSV ready for retention outreach.

## Dataset

Three input files (not included â€” see [BCG Data Science Virtual Program](https://www.theforage.com/)):

| File | Description |
|---|---|
| `ml_case_training_data.csv` | Client features (consumption, contract, margin) |
| `ml_case_training_hist_data.csv` | Monthly price history per client |
| `ml_case_training_output.csv` | Churn labels |

## Notebook structure

1. **EDA** â€” churn rate, categorical breakdowns, consumption distributions, tenure analysis
2. **Feature engineering** â€” price volatility, 6-month price delta, months to contract end/renewal, peak vs off-peak pricing features
3. **Modelling** â€” Logistic Regression, Random Forest, XGBoost, LightGBM compared on ROC-AUC with stratified train/test split
4. **Class imbalance** â€” SMOTE oversampling inside an imbalanced-learn pipeline
5. **Cross-validation** â€” 5-fold stratified CV on the best model
6. **SHAP explainability** â€” global feature importance, beeswarm direction plot, single-customer waterfall
7. **Export** â€” predictions saved to `churn_predictions.csv` with churn probability scores

## Results

| Metric | Value |
|---|---|
| Best model | LightGBM |
| Probability threshold | 0.20 |
| Recall (churners caught) | 42% |
| Precision (flags correct) | 26% |

The low precision reflects limited behavioural features in the dataset. The model is most useful when customers are ranked by churn probability and retention budget is focused on the top tier.

## Top churn drivers (SHAP)

- Gross and net power margin
- Customer origin segment
- Annual and last-month energy consumption
- Peak price volatility and range
- Months as a customer / years of tenure

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm shap
```

## Usage

1. Place the three CSV files in `~/Downloads/archive-2/` (or update `BASE_PATH` in the notebook)
2. Run all cells top to bottom
3. Predictions export to `~/Downloads/churn_predictions.csv`

## Notes

Model performance is constrained by available features. Adding behavioural signals â€” customer service interactions, payment history, or app engagement â€” would likely push precision above 35%. The current output is best used as a prioritisation tool, not a binary classifier.
