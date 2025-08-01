import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE
from fastapi import FastAPI
import uvicorn

# Step 1: Data Collection (Assuming Kaggle Credit Card Fraud Detection dataset)
data = pd.read_csv("creditcard.csv")

# Step 2: Data Preprocessing
def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Scale numerical features (excluding target 'Class')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(columns=['Class']))
    df_scaled = pd.DataFrame(scaled_features, columns=df.drop(columns=['Class']).columns)
    df_scaled['Class'] = df['Class']
    
    return df_scaled, scaler

# Step 3: Feature Engineering
def engineer_features(df):
    # Add time-based features
    df['Hour'] = df['Time'] // 3600 % 24
    df['Transaction_Frequency'] = df.groupby('Time')['Amount'].transform('count')
    return df

# Step 4: Handle Imbalanced Data
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Step 5: Model Development
def train_models(X_train, y_train):
    # Supervised: XGBoost
    xgb_model = XGBClassifier(random_state=42, scale_pos_weight=100)
    xgb_model.fit(X_train, y_train)
    
    # Unsupervised: Isolation Forest
    iso_forest = IsolationForest(contamination=0.001, random_state=42)
    iso_forest.fit(X_train)
    
    return xgb_model, iso_forest

# Step 6: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_pred)
    return precision, recall, f1, auc

# Step 7: FastAPI Deployment
app = FastAPI()

@app.post("/predict_fraud")
async def predict_fraud(transaction: dict):
    # Preprocess input transaction
    transaction_df = pd.DataFrame([transaction])
    transaction_scaled = scaler.transform(transaction_df)
    
    # Predict using XGBoost
    prediction = xgb_model.predict(transaction_scaled)[0]
    return {"fraud_probability": float(prediction)}

# Main Pipeline
if __name__ == "__main__":
    # Preprocess and engineer features
    data_scaled, scaler = preprocess_data(data)
    data_engineered = engineer_features(data_scaled)
    
    # Split data
    X = data_engineered.drop(columns=['Class'])
    y = data_engineered['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Balance data
    X_train_resampled, y_train_resampled = balance_data(X_train, y_train)
    
    # Train models
    xgb_model, iso_forest = train_models(X_train_resampled, y_train_resampled)
    
    # Evaluate
    precision, recall, f1, auc = evaluate_model(xgb_model, X_test, y_test)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
