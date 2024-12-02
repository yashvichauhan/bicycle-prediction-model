#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:20:06 2024

@author: yashvichauhan
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import pickle
import joblib


#Load data
path = "/Users/yashvichauhan/Centennial/Fall2024/COMP309/ProjectGroup2/"

filename = 'Bicycle_Thefts_Open_Data_-6397344066137313418.csv'

fullpath = os.path.join(path,filename)

df = pd.read_csv(fullpath)

#Keeping only necessary columns for prediction
df=df[['PRIMARY_OFFENCE','OCC_DOW','REPORT_DOW', 'HOOD_158','BIKE_MAKE','BIKE_TYPE', 'BIKE_COLOUR', 'BIKE_COST', 'STATUS', 'LOCATION_TYPE', 'PREMISES_TYPE']]

#Remove 'Unknown' status
df = df[df.STATUS != 'UNKNOWN'] 

#Manage Imbalanced class
df_majority = df[df.STATUS=='STOLEN']
df_minority = df[df.STATUS=='RECOVERED']

df_minority_upsampled = resample(df_minority,
                                 replace=True,     
                                 n_samples=36179,    
                                 random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

df_upsampled['STATUS'].value_counts()

#Fill in null values in the columns using most frequent value in that column
df_upsampled['BIKE_COLOUR'] = df_upsampled['BIKE_COLOUR'].fillna(df['BIKE_COLOUR'].mode().iloc[0])
df_upsampled['BIKE_COST'] = df_upsampled['BIKE_COST'].fillna(df['BIKE_COST'].mode().iloc[0])
df_upsampled['BIKE_MAKE'] = df_upsampled['BIKE_MAKE'].fillna(df['BIKE_MAKE'].mode().iloc[0])

#----------------------------------------Train test split and feature selection-------------------------------------------------

# making sure there's no null values remaining
df_upsampled.isnull().sum()

# Separate features and target
x = df_upsampled.drop('STATUS', axis=1)
y = np.where(df_upsampled['STATUS'] == 'STOLEN', 0, 1)

# Standardize numerical features
scaler = StandardScaler()
x_num = scaler.fit_transform(x[['BIKE_COST']].values.reshape(-1, 1))

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# One-hot encode categorical features
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
x_cat = enc.fit_transform(x[['PRIMARY_OFFENCE', 'OCC_DOW', 'REPORT_DOW', 'HOOD_158', 
                             'BIKE_MAKE', 'BIKE_TYPE', 'BIKE_COLOUR', 'LOCATION_TYPE', 'PREMISES_TYPE']])

# Save the encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(enc, f)

x_transformed = np.hstack((x_num, x_cat))
     
# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
     

# Feature names
num_features = ['BIKE_COST']
cat_features = enc.get_feature_names_out(['PRIMARY_OFFENCE', 'OCC_DOW', 'REPORT_DOW', 'HOOD_158', 
                                          'BIKE_MAKE', 'BIKE_TYPE', 'BIKE_COLOUR', 'LOCATION_TYPE', 'PREMISES_TYPE'])
feature_names = np.concatenate((num_features, cat_features))

joblib.dump(feature_names, '/Users/yashvichauhan/Centennial/Fall2024/COMP309/ProjectGroup2/model_columns_group2.pkl')

#---------------------------Decision tree---------------------------
dt_bicycle = DecisionTreeClassifier(criterion='entropy',max_depth=12, min_samples_split=40,
    min_samples_leaf=5, random_state=42)

dt_bicycle.fit(x_train, y_train)

importances = dt_bicycle.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(feature_importance_df)

sorted_indices = np.argsort(importances)[-10:]  # Top 10 important features

#plot the importances
plt.figure(figsize=(8, 6))
plt.title("Important Features using DecisionTreeClassifier")
plt.barh(range(len(sorted_indices)), importances[sorted_indices], color='g', align='center')
plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.show()

# Predictions with Decision Tree
y_pred_dt = dt_bicycle.predict(x_test)
y_pred_proba_dt = dt_bicycle.predict_proba(x_test)[:, 1]

# Decision Tree Accuracy sEvaluation
print("\nDecision Tree Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

#-------------------------Logistic Regression-----------------------------------

# Train Logistic Regression on Selected Features
x_train_selected = x_train[:, sorted_indices]
x_test_selected = x_test[:, sorted_indices]

# Train Logistic Regression on the selected features
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(x_train, y_train)

# Predictions with Logistic Regression
y_pred_lr = log_reg.predict(x_test)
y_pred_proba_lr = log_reg.predict_proba(x_train)[:, 1]

# Logistic Regression Accuracy Evaluation
print("\nLogistic Regression Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

#------------------------------------Random Forest Classifier-----------------------

# Initialize the Random Forest Classifier
rf_bicycle = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=20,
    min_samples_leaf=5, random_state=42)

# Train the Random Forest on the top features
rf_bicycle.fit(x_train, y_train)

# Predict with Random Forest Classification
y_pred_rf = rf_bicycle.predict(x_test)
y_pred_proba_rf = rf_bicycle.predict_proba(x_test)[:, 1]

print("\nRandom Forest Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

pickle.dump(rf_bicycle, open("rf_model.pkl", "wb"))


#-----------------------------Evaluation of different models----------------------------

# Evaluation Metrics for All Models
models = {
    "Decision Tree": (y_pred_dt, y_pred_proba_dt),
    "Logistic Regression": (y_pred_lr, y_pred_proba_lr),
    "Random Forest": (y_pred_rf, y_pred_proba_rf)
}

results = []
plt.figure(figsize=(10, 8))

for model_name, (y_pred, y_pred_proba) in models.items():
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{model_name} Confusion Matrix:\n{cm}")
    
    # Save confusion matrix as image
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stolen", "Recovered"], yticklabels=["Stolen", "Recovered"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(f"{model_name}_Confusion_Matrix.png")
    plt.show()
    
    # Classification Report
    report = classification_report(y_test, y_pred)
    print(f"\n{model_name} Classification Report:\n{report}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Append metrics for comparison
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc
    })

# ROC Curve
plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

# Generate ROC Curve for all models in one plot
plt.figure(figsize=(10, 8))

# Add diagonal for random chance
plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")

# Finalize the ROC plot
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for All Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Results Summary
results_df = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
print("\nModel Evaluation Summary:")
print(results_df)


# ------------------------- Save the Best Model -------------------------

# Load the model to ensure it's saved correctly
loaded_model = pickle.load(open("rf_model.pkl", "rb"))

# Test the loaded model with the test dataset
y_pred_loaded = loaded_model.predict(x_train)
y_pred_proba_loaded = loaded_model.predict_proba(x_train)[:, 1]

# Evaluate the Loaded Model
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
fpr_loaded, tpr_loaded, _ = roc_curve(y_test, y_pred_proba_loaded)
roc_auc_loaded = auc(fpr_loaded, tpr_loaded)

print("\nLoaded Model Evaluation:")
print(f"Accuracy: {accuracy_loaded:.4f}")
print(f"ROC AUC: {roc_auc_loaded:.4f}")

# ------------------------- Test with New Data (Deployment) -------------------------

# Load the saved scaler, encoder, model, and columns
with open("scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    loaded_encoder = pickle.load(f)

with open("model_columns_group2.pkl", "rb") as f:
    model_columns = joblib.load(f)

# Example input for testing the deployed model
test_df = pd.DataFrame([['THEFT UNDER', 'Friday', 'Wednesday', 168, 'MT', 'RG', 'BLK', 1300, 
                         'Single Home, House (Attach Garage, Cottage, Mobile), Corp. Bldg)', 'Outside']],
                       columns=['PRIMARY_OFFENCE', 'OCC_DOW', 'REPORT_DOW', 'HOOD_158',
                                'BIKE_MAKE', 'BIKE_TYPE', 'BIKE_COLOUR', 'BIKE_COST', 'LOCATION_TYPE', 'PREMISES_TYPE'])

# Check for missing values in the test data
test_df.isnull().sum()

# Transform the test data using the same pre-trained encoders and scalers
# Scale numerical features
test_num = loaded_scaler.fit_transform(test_df[['BIKE_COST']])  # Use .transform instead of .fit_transform

# One-hot encode categorical features
test_cat = loaded_encoder.fit_transform(test_df[['PRIMARY_OFFENCE', 'OCC_DOW', 'REPORT_DOW', 'HOOD_158',
                                             'BIKE_MAKE', 'BIKE_TYPE', 'BIKE_COLOUR', 'LOCATION_TYPE', 'PREMISES_TYPE']])

# Combine numerical and categorical features
test_transformed = np.hstack((test_num, test_cat))

# Create a DataFrame from the transformed data
transformed_test_df = pd.DataFrame(test_transformed)

# Align transformed data with model columns
# Add column names to the transformed data
transformed_test_df.columns = list(loaded_encoder.get_feature_names_out(
    ['PRIMARY_OFFENCE', 'OCC_DOW', 'REPORT_DOW', 'HOOD_158', 
     'BIKE_MAKE', 'BIKE_TYPE', 'BIKE_COLOUR', 'LOCATION_TYPE', 'PREMISES_TYPE']
)) + ['BIKE_COST']

# Reindex the DataFrame to match model_columns
aligned_test_df = transformed_test_df.reindex(columns=model_columns, fill_value=0)

# Ensure the shape matches the model's expectation
print(f"Transformed test data shape: {aligned_test_df.shape}")
print(f"Model expects: {loaded_model.n_features_in_} features")

# Predict using the loaded model
test_prediction = loaded_model.predict(aligned_test_df)
test_prediction_proba = loaded_model.predict_proba(aligned_test_df)

# Output results for the test case
print("\nDeployment Test Prediction Results:")
print(f"Predicted Class: {'Recovered' if test_prediction[0] == 1 else 'Stolen'}")
print(f"Prediction Probability: {test_prediction_proba[0]}")




