<h1 align="center">🏠 Home Credit Default Risk - Machine Learning Solution</h1>

<p align="center">
  <b>Predicting loan default risk using advanced ML techniques on Home Credit data</b>
  <br>
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
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
</ul>

<h2 id="overview">📖 Project Overview</h2>
<p>
  <b>Goal:</b> Predict whether a client will repay their home credit loan or default, using features from application data and credit history.<br>
  This notebook demonstrates:<br>
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
    <a href="https://www.kaggle.com/competitions/home-credit-default-risk/data" target="_blank">Download the Home Credit Default Risk data from Kaggle</a> and place the CSV files in your Google Drive or local path.
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
  <li><b>application_test.csv:</b> Test data for prediction</li>
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
</ul>

<h2 id="results">📊 Results</h2>
<ul>
  <li>Reports model performance: accuracy, precision, recall, F1, ROC-AUC, confusion matrix</li>
  <li>Provides insights into key features and missing data handling</li>
</ul>

<h2 id="insights">💡 Insights</h2>
<ul>
  <li>Test/train data proportion is balanced</li>
  <li>Many columns have significant missing data and require careful removal</li>
  <li><b>Target label:</b> 0 = loan repaid, 1 = loan not repaid (important for interpreting results and metrics)</li>
</ul>

<h2 id="contributing">🤝 Contributing</h2>
<p>
  Contributions are welcome! Please open an issue or pull request for discussion.
</p>

<h2 id="license">📝 License</h2>
<p>
  This project is licensed under the <a href="LICENSE">MIT License</a>.
</p>

<hr>
<p align="center">
  <i>Author: <a href="https://github.com/CodeBy-Omar">Omar</a></i>
</p>
