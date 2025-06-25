import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
# Feature data (methylation) - transposed later
df_training_features = pd.read_csv("df_training.csv")
df_test_features = pd.read_csv("df_test.csv")

# Metadata (including gestational age)
ano_training = pd.read_csv("ano_training.csv")
ano_test = pd.read_csv("ano_test.csv")

# --- Data Preprocessing and Merging ---
# Transpose feature data so samples are rows and features are columns
df_training_features_T = df_training_features.set_index('Unnamed: 0').T.reset_index()
df_test_features_T = df_test_features.set_index('Unnamed: 0').T.reset_index()
df_training_features_T.rename(columns={'index': 'Sample_ID'}, inplace=True)
df_test_features_T.rename(columns={'index': 'Sample_ID'}, inplace=True)

# Merge feature data with metadata using Sample_ID
train_df = pd.merge(ano_training, df_training_features_T, on='Sample_ID')
test_df = pd.merge(ano_test, df_test_features_T, on='Sample_ID')

# --- Prepare Data for Modeling ---
# Target variable: gestational_age
y_train = train_df['gestational_age']
y_test = test_df['gestational_age']

# Feature data: Methylation features (all columns except metadata and Sample_ID)
feature_cols = df_training_features_T.columns.drop('Sample_ID') # Get feature names from transposed data
X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# --- Feature Scaling ---
# Scale methylation features using StandardScaler (important for Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
# Using Ridge Regression (L2 regularization) - suitable for high-dimensional data
ridge_model = Ridge()
ridge_model.fit(X_train_scaled, y_train)

# --- Prediction on Test Set ---
y_pred = ridge_model.predict(X_test_scaled)

# --- Evaluation ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.4f}")

# --- Scatter Plot of Predicted vs Actual GA ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Gestational Age")
plt.ylabel("Predicted Gestational Age")
plt.title("Predicted vs Actual Gestational Age (Test Set)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--') # Diagonal line for perfect prediction
plt.legend([f"RMSE = {rmse:.4f}"], loc='upper left') # Include RMSE in legend
plt.grid(True)
plt.tight_layout()
plt.savefig("./out2/gemini-3.py.pdf", format="pdf")
