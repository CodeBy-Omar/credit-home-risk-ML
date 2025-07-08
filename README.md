<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
</head>
<body>
  <h1 align="center">🏠 Home Credit Default Risk - Machine Learning Solution</h1>
  <p align="center">
    <b>Predicting loan default risk using advanced ML techniques on Home Credit data</b><br>
    <a href="https://github.com/CodeBy-Omar/credit-home-risk-ML/blob/main/home_credit_risk.ipynb">
      <img src="https://img.shields.io/badge/Colab-Notebook-blue?logo=googlecolab">
    </a>
  </p>
  <hr>
  <h2>📋 Table of Contents</h2>
  <ul>
    <li><a href="#overview">Project Overview</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#requirements">Installation & Requirements</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#eda">EDA & Feature Engineering</a></li>
    <li><a href="#modeling">Modeling</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#insights">Insights</a></li>
  </ul>

  <h2 id="overview">📖 Project Overview</h2>
  <p>
    <b>Goal:</b> Predict whether a client will repay their home credit loan or default, using features from application data and credit history.<br>
    This notebook demonstrates:
    <ul>
      <li>Loading and cleaning data from Google Drive</li>
      <li>Exploratory Data Analysis (EDA)</li>
      <li>Feature engineering and missing value handling</li>
      <li>Model training with LightGBM, XGBoost, and CatBoost</li>
      <li>Performance evaluation and insights</li>
    </ul>
  </p>

  <h2 id="features">✨ Features</h2>
  <ul>
    <li>Automated data loading and preprocessing</li>
    <li>EDA with missing value analysis and visualization</li>
    <li>Advanced feature selection/removal logic</li>
    <li>Imbalanced class handling (RandomOverSampler)</li>
    <li>Training with LightGBM, XGBoost, CatBoost</li>
    <li>Model validation and detailed reporting</li>
  </ul>

  <h2 id="requirements">⚙️ Installation & Requirements</h2>
  <p>
    <b>Main requirements:</b>
    <ul>
      <li>Python 3.7+</li>
      <li>Jupyter Notebook or Google Colab</li>
      <li>pandas, numpy, matplotlib, seaborn</li>
      <li>scikit-learn</li>
      <li>lightgbm, xgboost, catboost</li>
    </ul>
    Install dependencies:
  </p>
  <pre>
pip install lightgbm xgboost catboost pandas numpy matplotlib seaborn scikit-learn imbalanced-learn plotly graphviz
  </pre>

  <h2 id="usage">🚀 Usage</h2>
  <ol>
    <li>
      Clone this repository:<br>
      <pre>git clone https://github.com/CodeBy-Omar/credit-home-risk-ML.git</pre>
    </li>
    <li>
      <a href="https://www.kaggle.com/competitions/home-credit-default-risk/data" target="_blank">Download the Home Credit Default Risk data from Kaggle</a> and place the CSV files in your Google Drive.
    </li>
    <li>
      Open <code>home_credit_risk.ipynb</code> in Jupyter Notebook or Google Colab.
    </li>
    <li>
      Update the data paths as needed in the notebook.
    </li>
    <li>
      Run all cells to reproduce the analysis and results.
    </li>
  </ol>

  <h2 id="data">🗂️ Data</h2>
  <ul>
    <li><b>application_train.csv:</b> Training data with loan application details and target (0: repaid, 1: not repaid)</li>
    <li><b>application_test.csv:</b> Test data for prediction (<b>Note: The test set was not used for local model assessment, because its true labels were only available upon submission to Kaggle. Therefore, all model training, validation, and testing were performed using splits of the <code>application_train.csv</code> data. The <code>application_test.csv</code> was preprocessed in the same way as train, to demonstrate the data pipeline and show how predictions would be produced if test labels were accessible for final model evaluation.</b>)</li>
  </ul>
  <p>
    <i>Note: Only the main application files are used in this analysis. Other supplemental files can be integrated for feature enrichment.</i>
  </p>

  <h2 id="eda">🔎 EDA & Feature Engineering</h2>
  <ul>
    <li>Computes missing value percentages per feature</li>
    <li>Removes features with excessive missingness (&gt;50%)</li>
    <li>Special investigation of <code>AMT_REQ_CREDIT_BUREAU_*</code> columns</li>
    <li>Saves missing value summary to CSV for review</li>
  </ul>

  <h2 id="modeling">🤖 Modeling</h2>
  <ul>
    <li>Handles class imbalance with RandomOverSampler</li>
    <li>Trains LightGBM, XGBoost, and CatBoost classifiers</li>
    <li>Model selection via cross-validation and grid search</li>
    <li><b>Threshold Tuning:</b> Different classification thresholds were tested and a threshold of 0.4 gave the best balance between precision and recall.</li>
    <li><b>Feature Inclusion:</b> A base model was first trained using only the <code>application_train.csv</code> file. Then, features from <code>bureau.csv</code> were added. This improved the ROC-AUC from <b>0.7474</b> (base model) to <b>0.7512</b> (with bureau features).</li>
    <li>Further improvement is possible: When adding all available files in the competition (such as bureau_balance, credit_card_balance, etc.), the ROC-AUC can surpass <b>0.80+</b>.</li>
  </ul>

  <h2 id="results">📊 Results</h2>
  <ul>
    <li>Reports model performance: accuracy, precision, recall, F1, ROC-AUC, confusion matrix</li>
    <li>Provides insights into key features and missing data handling</li>
    <li><b>Key Score:</b> ROC-AUC reached <b>0.7512</b> with application and bureau features (vs. 0.7474 for base), and can exceed 0.80 with full feature integration.</li>
  </ul>

  <h2 id="insights">💡 Insights</h2>
  <ul>
    <li>Test/train data proportion is not balanced</li>
    <li>Many columns have significant missing data and require careful removal</li>
    <li><b>Target label:</b> 0 = loan repaid, 1 = loan not repaid (important for interpreting results and metrics)</li>
    <li><b>About test set:</b> The <code>application_test.csv</code> was not used for local validation or model selection, as its ground truth labels are only available upon submission to Kaggle. All model development and evaluation were therefore based on splits of the <code>application_train.csv</code> data. The test set was preprocessed and included in the pipeline to illustrate how the workflow would handle unseen data and to demonstrate the approach for real-world deployment if test labels were accessible for final assessment.</li>
  </ul>
  <hr>
  <p align="center">
    <i>Author: <a href="https://github.com/CodeBy-Omar">Omar</a></i>
  </p>
</body>
</html>
