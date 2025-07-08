# 🏠 Home Credit Default Risk - Machine Learning Solution

**Predicting loan default risk using advanced ML techniques on Home Credit data**  
[![Colab-Notebook](https://img.shields.io/badge/Colab-Notebook-blue?logo=googlecolab)](https://github.com/CodeBy-Omar/credit-home-risk-ML/blob/main/home_credit_risk.ipynb)

---

## 📋 Table of Contents
- [Project Overview](#overview)
- [Features](#features)
- [Installation & Requirements](#requirements)
- [Usage](#usage)
- [Data](#data)
- [EDA & Feature Engineering](#eda)
- [Modeling](#modeling)
- [Results](#results)
- [Insights](#insights)

---

## 📖 Project Overview

**Goal:** Predict whether a client will repay their home credit loan or default, using features from application data and credit history.

This notebook demonstrates:
- Loading and cleaning data from Google Drive
- Exploratory Data Analysis (EDA)
- Feature engineering and missing value handling
- Model training with LightGBM, XGBoost, and CatBoost
- Performance evaluation and insights

> **Important:**  
> The modeling pipeline was performed in clear sequential steps:
> - **First**, all modeling, validation, and evaluation were performed **using only `application_train.csv`**. The base model was trained and assessed on this data split, with no external features.
> - **After obtaining baseline results**, I **preprocessed `bureau.csv`** and **integrated its features into `application_train.csv`**. The models were retrained and evaluated with these additional features to **explicitly measure the improvement** gained by incorporating credit history data.
> - This procedure makes it clear how much each data source contributes to the overall prediction performance.

---

## ✨ Features

- Automated data loading and preprocessing
- EDA with missing value analysis and visualization
- Advanced feature selection/removal logic
- Imbalanced class handling (RandomOverSampler)
- Training with LightGBM, CatBoost
- Model validation and detailed reporting

---

## ⚙️ Installation & Requirements

**Main requirements:**
- Python 3.7+
- Jupyter Notebook or Google Colab
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- lightgbm, catboost

Install dependencies:
```sh
pip install lightgbm xgboost catboost pandas numpy matplotlib seaborn scikit-learn imbalanced-learn plotly graphviz
```

---

## 🚀 Usage

1. **Clone this repository:**
   ```sh
   git clone https://github.com/CodeBy-Omar/credit-home-risk-ML.git
   ```
2. [Download the Home Credit Default Risk data from Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place the CSV files in your Google Drive.
3. Open `home_credit_risk.ipynb` in Jupyter Notebook or Google Colab.
4. Update the data paths as needed in the notebook.
5. Run all cells to reproduce the analysis and results.

---

## 🗂️ Data

- **application_train.csv:** Training data with loan application details and the target label (0: repaid, 1: not repaid)
- **application_test.csv:** Test data for prediction  
  > *Note: The test set was not used for local model assessment because its true labels are only available upon submission to Kaggle. All model training, validation, and testing were performed using splits of `application_train.csv`. The `application_test.csv` was preprocessed in the same way as train to demonstrate the data pipeline and show how predictions would be produced if test labels were accessible for final model evaluation.*

> **Data integration process:**  
> Only the main application files are used in this analysis by default. **After obtaining results with `application_train.csv` alone, `bureau.csv` was carefully preprocessed and merged with the train features. The subsequent improvement was measured and reported in the results section.**  
> Other supplemental files can be integrated for further feature enrichment.

---

## 🔎 EDA & Feature Engineering

- Computes missing value percentages per feature
- Removes features with excessive missingness (>50%)
- Special investigation of `AMT_REQ_CREDIT_BUREAU_*` columns
- Saves missing value summary to CSV for review

---

## 🤖 Modeling

- Handles class imbalance with RandomOverSampler
- Trains LightGBM, XGBoost, and CatBoost classifiers
- Model selection via cross-validation and grid search
- **Threshold Tuning:** Different classification thresholds were tested and a threshold of 0.4 gave the best balance between precision and recall.
- **Clear Data Addition Process:**
  - **Base Model:** First trained on `application_train.csv` only.
  - **Feature Addition:** Then, after initial results, features from `bureau.csv` were added to `application_train.csv` and the models were retrained.
  - This **clear, step-by-step approach** allows for a transparent assessment of the impact of additional data sources.
- **Performance:** ROC-AUC improved from **0.7474** (base) to **0.7512** (with bureau.csv features).  
  Further improvement is possible: When adding all available competition files (such as bureau_balance, credit_card_balance, etc.), the ROC-AUC can surpass **0.80+**.

---

## 📊 Results

- Reports model performance: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Provides insights into key features and missing data handling
- **Key Score:** ROC-AUC reached **0.7512** with application and bureau features (vs. 0.7474 for base), and can exceed 0.80 with full feature integration.

---

## 💡 Insights

- Test/train data proportion is not balanced
- Many columns have significant missing data and require careful removal
- **Target label:** 0 = loan repaid, 1 = loan not repaid (important for interpreting results and metrics)
- **About test set:** The `application_test.csv` was not used for local validation or model selection, as its ground truth labels are only available upon submission to Kaggle. All model development and evaluation were therefore based on splits of the `application_train.csv` data. The test set was preprocessed and included in the pipeline to illustrate how the workflow would handle unseen data and to demonstrate the approach for real-world deployment if test labels were accessible for final assessment.

---

<p align="center">
  <i>Author: <a href="https://github.com/CodeBy-Omar">Omar</a></i>
</p>
