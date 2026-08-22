# Fraud Detection Analysis

A Python machine learning project that analyzes transaction data and builds a classification model to identify potentially fraudulent transactions.

The project uses data stored in a SQLite database and applies data cleaning, preprocessing, exploratory analysis, and logistic regression with Scikit-learn.

## Project Overview

Fraud detection is a classification problem where the goal is to distinguish legitimate transactions from fraudulent ones.

This project demonstrates an end-to-end machine learning workflow including:

- Loading data from a SQLite database
- Cleaning and preparing transaction data
- Handling missing values
- Processing numeric and categorical variables
- Splitting data into training and testing sets
- Building a Scikit-learn preprocessing and modeling pipeline
- Training a logistic regression model
- Evaluating model performance
- Examining the variables that influence fraud predictions

## Technologies Used

- Python
- SQLite
- Pandas
- NumPy
- Scikit-learn

## Machine Learning Workflow

### 1. Data Loading

Transaction data is retrieved from a SQLite database using Python's `sqlite3` module and loaded into a Pandas DataFrame.

### 2. Data Cleaning

The dataset is cleaned before modeling by:

- Converting blank values to missing values
- Correcting column data types
- Separating the target variable from predictor variables
- Identifying numeric and categorical features

### 3. Preprocessing

Different preprocessing steps are applied depending on the type of feature.

Numeric variables:

- Missing values are imputed
- Features are standardized using `StandardScaler`

Categorical variables:

- Missing values are imputed
- Categories are converted into numeric features using `OneHotEncoder`

A `ColumnTransformer` combines these transformations into a single preprocessing step.

### 4. Model

The project uses logistic regression to estimate the probability that a transaction is fraudulent.

The preprocessing steps and logistic regression model are combined using a Scikit-learn `Pipeline`.

This ensures that the same transformations applied during training are also applied when the model makes predictions.

### 5. Model Evaluation

Model performance is evaluated using metrics including:

- Confusion matrix
- Precision
- Recall
- F1-score
- Classification report

For fraud detection, recall for the fraud class is especially important because it measures how many actual fraudulent transactions the model successfully identifies.

At the same time, precision is important because incorrectly flagging too many legitimate transactions can create unnecessary investigations or customer friction.

## Model Interpretation

Because logistic regression is an interpretable model, its coefficients can be examined to understand which features have the strongest relationship with predicted fraud.

Positive coefficients indicate that a feature is associated with a higher predicted probability of fraud, while negative coefficients indicate an association with a lower predicted probability.

The magnitude of the coefficient indicates the strength of that relationship.

This allows the model to be evaluated not only based on predictive performance but also on the factors driving its decisions.

## Project Structure

```text
Fraud-Detection/
│
├── Fraud.py
├── fraud.db
├── README.md
└── .gitignore
