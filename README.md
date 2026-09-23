# Credit-Risk-Classification
A machine learning project for predicting whether a loan applicant represents Good Credit Risk or Bad Credit Risk using demographic, financial, and loan-related features.

The project compares multiple classification algorithms and evaluates them using Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrices. Random Forest is further tuned using GridSearchCV, with Recall treated as the primary metric because identifying potentially risky borrowers is important in credit-risk applications.

📌 Project Overview

Credit risk classification is a binary classification problem in which the objective is to estimate whether a loan applicant is likely to be a good or bad credit risk.

This project implements an end-to-end machine learning workflow:

Data Generation/Collection → Exploratory Data Analysis → Preprocessing → Feature Engineering → Model Training → Hyperparameter Tuning → Model Evaluation → Feature Importance → Business Interpretation

Objective

Build and compare machine learning models that can classify loan applicants based on their credit-related characteristics.

Target
0 → Good Credit Risk
1 → Bad Credit Risk
📂 Project Structure
Credit-Risk-Classification/
│
├── Data Folder/
│   ├── german_credit_data.csv
│   ├── X_train.csv
│   └── X_test.csv
│
├── Models Folder/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── Notebook/
│   ├── EDA & Preprocessing.ipynb
│   ├── Model Development & Hyperparameter Tuning.ipynb
│   └── Model Evaluation & Comparison.ipynb
│
├── Visualizations Folder/
│   ├── target_distribution.png
│   ├── confusion_matrices (2).png
│   ├── roc_curves (2).png
│   └── feature_importance (2).png
│
├── Report/
│   └── Technical_Report_Dhruv.pdf
│
└── README.md
🛠️ Technologies Used
Python
Pandas — Data manipulation
NumPy — Numerical computation
Matplotlib — Data visualization
Seaborn — Statistical visualization
Scikit-learn — Machine learning
XGBoost — Gradient boosting classification
MLflow — Experiment tracking and model logging
Jupyter Notebook / Google Colab
📊 Dataset

The project uses credit-related applicant information containing features such as:

Feature	Description
Age	Applicant's age
Sex	Applicant's gender
Job	Job category
Housing	Housing status
Saving accounts	Savings account status
Checking account	Checking account status
Credit amount	Requested credit amount
Duration	Loan duration
Purpose	Purpose of the loan
Risk	Target variable

The project also performs preprocessing to handle missing values and categorical variables.

🔎 Exploratory Data Analysis

The project analyzes the distribution of credit-risk classes and examines relationships between applicant characteristics and risk.

Target Distribution

The distribution of the Risk variable is visualized to understand the number of good and bad credit-risk observations.

The project also generates visualizations for model evaluation and feature importance.

⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Missing Value Handling

Missing values in:

Saving accounts
Checking account

are replaced with the little category.

2. Feature Engineering

Additional features are created, including:

Credit_to_Duration_Ratio
Age_Group
Account_Stability

Account_Stability combines the applicant's saving and checking account categories into a numerical stability measure.

3. Categorical Encoding

Categorical variables are converted into numerical representations using one-hot encoding.

pd.get_dummies(..., drop_first=True)
4. Target Encoding

The risk classes are converted into binary values:

Good → 0
Bad  → 1
5. Train-Test Split

The dataset is divided into:

80% → Training data
20% → Testing data

using a fixed random state for reproducibility.

🤖 Machine Learning Models

Four classification algorithms are implemented and compared.

1. Logistic Regression

A linear classification model used as a baseline.

LogisticRegression(max_iter=1000)
2. Decision Tree

A tree-based model that learns decision rules from the input features.

DecisionTreeClassifier(random_state=42)
3. Random Forest

An ensemble of decision trees designed to improve prediction performance and robustness.

RandomForestClassifier(random_state=42)
4. XGBoost

A gradient boosting algorithm that builds an ensemble of sequential decision trees.

XGBClassifier(random_state=42)
🎯 Hyperparameter Tuning

Random Forest is further optimized using GridSearchCV with 5-fold cross-validation.

The parameter grid includes:

{
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2]
}

The tuning process uses Recall as the scoring metric.

GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='recall'
)

The best model is then logged using MLflow.

📈 Model Evaluation

The models are evaluated using multiple classification metrics.

Accuracy

Measures the overall proportion of correct predictions.

Precision

Measures how many observations predicted as bad credit risk were actually bad risks.

Recall

Measures how many actual bad-risk applicants were successfully identified.

F1-Score

Provides a balance between Precision and Recall.

AUC-ROC

Measures the model's ability to distinguish between the two risk classes across different classification thresholds.

📊 Evaluation Visualizations

The project generates:

Confusion Matrices

Used to visualize:

True Positives
True Negatives
False Positives
False Negatives
ROC Curves

ROC curves are plotted for all four classification models and their AUC values are compared.

Feature Importance

The Random Forest model is used to identify the features that contribute most strongly to its predictions.

💼 Business Interpretation

In a credit-risk application, Recall is particularly important because failing to identify a genuinely risky applicant can result in financial losses.

Therefore, the project gives particular attention to reducing False Negatives.

The project evaluates the models not only from a statistical perspective but also from a business perspective, considering the potential cost of incorrect credit-risk decisions.

🧪 Experiment Tracking

MLflow is used to track the Random Forest hyperparameter-tuning experiment.

The experiment is named:

Credit_Risk_Classification_Dhruv

The best Random Forest parameters are logged along with the trained model.

💾 Saved Models

Pre-trained models are provided in the Models Folder:

logistic_regression.pkl
decision_tree.pkl
random_forest.pkl
xgboost.pkl

These models can be loaded using Python's joblib or pickle depending on how the model files were serialized.

🚀 How to Run the Project
1. Clone the Repository
git clone <your-repository-url>
cd Credit-Risk-Classification
2. Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost mlflow jupyter
3. Start Jupyter Notebook
jupyter notebook
4. Run the Notebooks in Order

Run the notebooks in the following sequence:

1. EDA & Preprocessing.ipynb
            ↓
2. Model Development & Hyperparameter Tuning.ipynb
            ↓
3. Model Evaluation & Comparison.ipynb
