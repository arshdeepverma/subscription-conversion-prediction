# Subscription Conversion Prediction

An end-to-end machine learning project that predicts whether a free user will convert into a paid subscriber based on their engagement behavior.

The project covers the complete machine learning workflow — from data cleaning and exploratory analysis to feature engineering, model comparison, cross-validation, hyperparameter tuning, and conversion probability prediction.

---

## Overview

Subscription-based businesses often have a large population of free users, but only a subset eventually converts to paid plans.

This project builds a machine learning classification pipeline to identify users with a higher predicted likelihood of subscription conversion.

The predictions can support use cases such as:

* Identifying high-potential free users
* Prioritizing conversion campaigns
* Personalizing offers and messaging
* Improving marketing efficiency
* Understanding engagement signals associated with conversion

> **Note:** This project uses a small synthetic dataset intended for learning and portfolio demonstration rather than production deployment.

---

## Problem Statement

### Business Problem

Given a free user's engagement and usage behavior, can we predict whether that user will convert into a paid subscriber?

### Machine Learning Problem

This is a **supervised binary classification** problem.

### Target Variable

`Converted`

| Value | Meaning                       |
| ----- | ----------------------------- |
| `0`   | User did not convert          |
| `1`   | User became a paid subscriber |

---

## Dataset Features

The model uses user demographic, engagement, trial, and support-related attributes.

| Feature                | Description                          |
| ---------------------- | ------------------------------------ |
| `Age`                  | User age                             |
| `Days_Since_Signup`    | Number of days since registration    |
| `Sessions`             | Number of sessions                   |
| `Visits`               | Number of visits                     |
| `Features_Used`        | Number of product features used      |
| `Avg_Session_Minutes`  | Average session duration             |
| `Trial_Days_Used`      | Number of trial days used            |
| `Support_Interactions` | Number of support interactions       |
| `Sessions_Per_Day`     | Sessions relative to signup duration |
| `Visits_Per_Session`   | Visits relative to sessions          |
| `Features_Per_Session` | Features used relative to sessions   |
| `Engagement_Score`     | Composite engagement metric          |

`User_ID` is excluded from modeling because it is an identifier rather than a predictive feature.

---

## Machine Learning Pipeline

```text
Raw User Data
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Train / Test Split
      │
      ├──────────────┐
      ▼              ▼
Logistic Regression  Decision Tree
      │              │
      └──────┬───────┘
             ▼
       Random Forest
             │
             ▼
       Model Comparison
             │
             ▼
     Cross-Validation
             │
             ▼
 Hyperparameter Tuning
             │
             ▼
       Final Model
             │
             ▼
Conversion Probability
```

---

## Approach

### 1. Data Cleaning

The preprocessing workflow includes:

* Removing duplicate records
* Detecting missing numerical values
* Imputing missing numerical values using the median

---

### 2. Exploratory Data Analysis

EDA is performed to understand:

* Subscription conversion distribution
* Average engagement metrics by conversion status
* Feature distributions across converted and non-converted users
* Relationships between engagement behavior and conversion

Visual analysis includes target-distribution plots and feature-level boxplots.

---

### 3. Feature Engineering

Additional behavioral features are created to provide the models with more meaningful engagement signals:

```text
Sessions_Per_Day
Visits_Per_Session
Features_Per_Session
Engagement_Score
```

The composite engagement score is calculated from sessions, visits, features used, and trial days.

---

## Models

Three classification algorithms are trained and compared.

### Logistic Regression

Used as an interpretable baseline model for binary classification.

A `StandardScaler` is incorporated into a Scikit-learn pipeline because Logistic Regression benefits from features being on comparable scales.

### Decision Tree

A Decision Tree is used to capture non-linear relationships and rule-based decision patterns.

### Random Forest

Random Forest is used as an ensemble model consisting of multiple decision trees.

The initial configuration uses:

* `n_estimators = 300`
* `max_depth = 8`
* `min_samples_split = 5`
* `class_weight = "balanced"`

---

## Model Evaluation

Models are evaluated using multiple classification metrics rather than relying on accuracy alone.

### Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report
* ROC Curve

### Cross-Validation

Because the dataset contains only 100 users, the project also performs **5-fold stratified cross-validation** using ROC-AUC as the scoring metric.

This provides a more robust comparison than relying solely on a single train/test split.

---

## Hyperparameter Optimization

Random Forest hyperparameters are optimized using `GridSearchCV`.

The search explores:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

The optimization objective is **ROC-AUC**, with 5-fold stratified cross-validation.

The goal is to identify a model configuration that generalizes better rather than simply maximizing performance on one test set.

---

## Model Interpretability

Random Forest feature importance is calculated to identify which features contributed most to the model's predictions.

> Feature importance indicates association with model predictions; it does not establish that a feature causally drives subscription conversion.

---

## Prediction Output

The final workflow generates a conversion probability for each user.

Example output structure:

| Feature                  | Description                              |
| ------------------------ | ---------------------------------------- |
| `Actual_Converted`       | Actual conversion outcome                |
| `Conversion_Probability` | Estimated probability of conversion      |
| `Predicted_Converted`    | Binary prediction using a 0.50 threshold |

Users can then be sorted according to their predicted conversion probability.

---

## Business Application

The model output can support customer segmentation and experimentation.

For example:

**High predicted probability**

Potentially prioritize for conversion-focused messaging.

**Medium predicted probability**

Test personalized offers, onboarding improvements, or engagement campaigns.

**Low predicted probability**

Focus on improving engagement and onboarding before aggressively targeting conversion.

These actions are hypotheses for business experimentation, not guarantees of user behavior.

---

## Project Structure

```text
subscription-conversion-prediction/
│
├── data/
│   └── subscription_conversion_100_users.csv
│
├── notebooks/
│   └── Subscription_Conversion_Prediction_ML_Project.ipynb
│
├── models/
│   └── README.md
│
├── src/
│   └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The current project is primarily implemented in a Jupyter/Google Colab notebook. The additional folders represent a recommended structure for evolving the project into a more production-oriented repository.

---

## Tech Stack

**Language**

* Python

**Data Analysis**

* Pandas
* NumPy

**Visualization**

* Matplotlib

**Machine Learning**

* Scikit-learn

**Development Environment**

* Jupyter Notebook
* Google Colab

---

## Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Binary classification
* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Stratified train/test splitting
* Feature scaling
* Logistic Regression
* Decision Trees
* Random Forest
* Model comparison
* Cross-validation
* Hyperparameter optimization
* ROC-AUC analysis
* Confusion matrices
* Feature importance
* Probability-based predictions
* Business interpretation of ML outputs

---

## Limitations

This project should be interpreted as a **portfolio and learning implementation**, not a production-ready prediction system.

The dataset contains only 100 users, meaning that model performance can vary substantially depending on the train/test split.

A production implementation would require:

* A significantly larger historical dataset
* Time-based validation where appropriate
* Data leakage checks
* Model calibration
* Robust feature pipelines
* Production monitoring
* Drift detection
* Periodic model retraining
* Business-defined decision thresholds
* Evaluation on real-world outcomes

---

## Future Improvements

Potential extensions include:

* Increasing the training dataset size
* Adding real subscription and behavioral data
* Implementing automated preprocessing pipelines
* Testing additional classification algorithms
* Performing feature selection
* Calibrating predicted probabilities
* Optimizing the classification threshold based on business costs
* Adding SHAP-based model explainability
* Building a Streamlit prediction interface
* Saving the trained model with `joblib`
* Creating an inference API
* Adding automated tests and CI/CD
* Monitoring model performance after deployment

---

## Reproducibility

The notebook uses fixed random seeds where applicable to make the experiments more reproducible.

The train/test split uses:

```text
test_size = 0.20
random_state = 42
stratify = y
```

Cross-validation uses a 5-fold `StratifiedKFold` configuration with a fixed random state.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/subscription-conversion-prediction.git
cd subscription-conversion-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/Subscription_Conversion_Prediction_ML_Project.ipynb
```

Alternatively, the notebook can be executed using Google Colab.

---

## Requirements

Recommended `requirements.txt`:

```text
pandas
numpy
matplotlib
scikit-learn
jupyter
```

---

## Results

Model performance is calculated directly within the notebook using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Cross-validation ROC-AUC

The repository intentionally does not hard-code performance numbers in this README. This ensures that reported results remain tied to the actual notebook execution rather than presenting unverified metrics.

---

## Author

**Arsh**

Machine Learning | Data Analytics | Business Intelligence

---

## License

This project is intended for educational and portfolio purposes.

If you plan to distribute or adapt the project, add an appropriate open-source license such as MIT.
