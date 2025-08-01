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



Key Notes:

Dataset: The Kaggle Credit Card Fraud Detection dataset is anonymized, with PCA-transformed features and a 'Class' column (0 for legitimate, 1 for fraud). Synthetic data can be generated using libraries like SDV if needed.
Imbalanced Data: SMOTE oversamples the minority class (fraud) to improve model performance.
Evaluation Metrics: Precision and recall are critical due to the rarity of fraud cases. AUC-ROC ensures robust performance evaluation.
Deployment: FastAPI enables real-time predictions, suitable for payment systems. Monitoring can be added using Prometheus or custom drift detection.
