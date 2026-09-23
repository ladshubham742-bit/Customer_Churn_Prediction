Customer Churn Prediction Using Machine Learning

📌 Project Overview

This project uses Machine Learning classification algorithms to predict whether a customer is likely to churn.

The notebook covers the complete workflow from data understanding and exploratory data analysis (EDA) to preprocessing, model training, evaluation, visualization, and feature-importance analysis.

🎯 Objective

The main objectives of this project are to:

Analyze customer demographic, usage, support, payment, subscription, and spending information.

Understand patterns associated with customer churn.

Prepare the data for machine learning.

Train and compare multiple classification models.

Evaluate model performance using standard classification metrics.

Identify the features that are most important for churn prediction.

🤖 Machine Learning Problem

This is a binary classification problem.

Churn Value

Meaning

0

Customer does not churn

1

Customer churns

The target variable is Churn.

🛠️ Technologies & Libraries

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Jupyter Notebook

Machine Learning Algorithms

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

📂 Project Structure

Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── data/
│   └── customer_churn_dataset-testing-master.csv
└── README.md

The notebook currently references the dataset using a local Windows path. Update the path when running the project on another computer.

🔄 Project Workflow

Dataset
   ↓
Load Data
   ↓
Understand Data
   ↓
Check Missing Values
   ↓
Check Duplicates
   ↓
Check Unique Values
   ↓
Target Variable Analysis
   ↓
Exploratory Data Analysis
   ↓
Remove CustomerID
   ↓
Encode Categorical Variables
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Train ML Models
   ├── Logistic Regression
   ├── Decision Tree
   └── Random Forest
   ↓
Model Evaluation
   ↓
Confusion Matrix
   ↓
ROC Curve & ROC-AUC
   ↓
Feature Importance

🔍 Data Exploration & EDA

The notebook performs several data-analysis steps:

Data Understanding

Dataset shape

Data types and information

Descriptive statistics

Missing-value analysis

Duplicate-row analysis

Unique-value analysis

Target Analysis

The distribution of the Churn target variable is visualized to understand the proportion of customers who churn and do not churn.

Numerical Analysis

The project explores relationships between churn and:

Age

Support Calls

Payment Delay

Usage Frequency

Categorical Analysis

The project analyzes churn according to:

Subscription Type

Contract Length

🧹 Data Preprocessing

1. Remove CustomerID

CustomerID is removed because it is an identifier rather than a customer-behavior feature.

X = df.drop(
    ["Churn", "CustomerID"],
    axis=1
)

y = df["Churn"]

2. Encode Categorical Variables

Categorical columns are converted into numerical features using one-hot encoding:

X = pd.get_dummies(
    X,
    drop_first=True
)

3. Train/Test Split

The dataset is divided into training and testing sets using an 80/20 split with stratification:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

4. Feature Scaling

StandardScaler is used for Logistic Regression:

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

🧠 Model Training

1. Logistic Regression

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

2. Decision Tree

The Decision Tree uses:

DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

3. Random Forest

The Random Forest uses:

RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

📊 Model Evaluation

The notebook compares the three models using:

Accuracy

Precision

Recall

F1 Score

It also creates:

Model-performance comparison chart

Random Forest confusion matrix

ROC curve

ROC-AUC score

Evaluation Function

def evaluate_model(name, y_true, predictions):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions),
        "Recall": recall_score(y_true, predictions),
        "F1 Score": f1_score(y_true, predictions)
    }

📈 ROC-AUC Analysis

The notebook calculates the ROC curve and ROC-AUC score for the Random Forest model.

rf_probability = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_probability
)

auc_score = roc_auc_score(
    y_test,
    rf_probability
)

⭐ Feature Importance

Random Forest feature importance is used to identify which processed features contribute most to the model's predictions.

The notebook creates a ranked feature-importance table and visualizes the Top 10 features.

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

🚀 How to Run the Project

1. Clone the Repository

git clone https://github.com/YOUR-USERNAME/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2. Install Required Libraries

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

3. Start Jupyter Notebook

jupyter notebook

4. Open the Notebook

Open:

Customer_Churn_Prediction.ipynb

5. Update the Dataset Path

Change the pd.read_csv() path in the notebook to match the location of your dataset.

For example:

df = pd.read_csv(
    "data/customer_churn_dataset-testing-master.csv"
)

6. Run All Cells

Run the notebook from top to bottom to reproduce the analysis, model training, evaluation, and visualizations.

📌 Key Project Outcomes

This project demonstrates an end-to-end machine learning workflow for a customer churn problem:

Data loading and inspection

Data-quality checks

Exploratory data analysis

Feature preparation

Categorical encoding

Train/test splitting

Feature scaling

Multiple classification algorithms

Model comparison

Confusion-matrix analysis

ROC-AUC analysis

Random Forest feature importance

Model performance values are intentionally not hard-coded in this README. Run the notebook on the dataset to generate the current evaluation results.

💼 Business Use Case

A customer churn prediction system can help organizations analyze customer behavior and identify patterns associated with customer attrition.

Potential applications include:

Customer retention analysis

Subscription analysis

Customer-support analysis

Payment-behavior analysis

Identifying important churn-related factors

Supporting data-driven retention strategies

🔮 Future Improvements

Possible extensions to this project include:

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV

Cross-validation

Handling class imbalance if present

Trying additional classification algorithms

Building an interactive dashboard using Power BI or Tableau

Creating a Streamlit web application

Saving the trained model with Joblib

Deploying the model as an API

Adding automated model monitoring

👨‍💻 Author

Shubham Lad

Artificial Intelligence & Data Science Student

Skills Demonstrated

Python · Machine Learning · Data Analysis · EDA · Pandas · Scikit-learn · Data Visualization

⭐ If You Find This Project Useful

Feel free to star ⭐ the repository and use the project as a reference for learning customer churn prediction and machine learning classification.
