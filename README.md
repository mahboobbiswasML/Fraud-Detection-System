# Fraud-Detection-System
This pipeline builds a real-time fraud detection system for credit card or online payment transactions. The goal is to classify transactions as fraudulent or legitimate using supervised and unsupervised learning techniques, handling the highly imbalanced nature of fraud data.



Pipeline Steps:

Data Collection: Use a dataset like Kaggle’s Credit Card Fraud Detection dataset or synthetic data for privacy.
Data Preprocessing: Handle missing values, encode categorical variables, scale numerical features, and address class imbalance using SMOTE.
Exploratory Data Analysis (EDA): Analyze transaction patterns, distributions, and correlations to identify fraud indicators.
Feature Engineering: Create features like transaction frequency, time-based aggregations, and anomaly scores.
Model Development: Train an ensemble model (e.g., XGBoost) and an unsupervised model (e.g., Isolation Forest) for comparison.
Model Evaluation: Use precision, recall, F1-score, and AUC-ROC to evaluate performance on imbalanced data.
Model Deployment: Deploy the model as a REST API using FastAPI for real-time predictions.
Monitoring: Implement a monitoring system to track model drift and retrain periodically.

Conclution: Created an end to end model which will detect fraud.
