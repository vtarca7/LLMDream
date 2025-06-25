import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load feature data and metadata for training set
def load_data_csv(csv_path):
    # Read the .csv file with features as rows and samples as columns
    df = pd.read_csv(csv_path, header=0).T
    return df

df_training_features = load_data_csv("df_training.csv")
ano_training = pd.read_csv("ano_training.csv")

# Merge feature data with metadata using Sample_ID
training_df = pd.merge(df_training_features, ano_training, on='Sample_ID')

# Split training data into features and target variable
X_train = training_df.drop(columns=['Gestational_Age'])
y_train = training_df['Gestational_Age']

# Load test set feature data and metadata
df_test_features = load_data_csv("df_test.csv")
ano_test = pd.read_csv("ano_test.csv")

# Merge test data with metadata using Sample_ID
test_df = pd.merge(df_test_features, ano_test, on='Sample_ID')

# Split test data into features and target variable (for prediction)
X_test = test_df.drop(columns=['Gestational_Age'])
y_true = test_df['Gestational_Age']

# Perform feature scaling if necessary
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split training data into train and validation sets for hyperparameter tuning
X_trainCV, X_val, y_trainCV, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42)

# Define parameter grid for Lasso regression
param_grid = {'alpha': [0.1, 1.0, 5.0, 10.0]}

# Perform grid search with cross-validation to find optimal alpha
lasso = Lasso()
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_trainCV, y_trainCV)

best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha found: {best_alpha}")

# Fit final model with optimal alpha
final_lasso = Lasso(alpha=best_alpha)
final_lasso.fit(X_train_scaled, y_train)

# Make predictions on test set
y_pred = final_lasso.predict(X_test_scaled)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.4f}")

# Create scatter plot of predicted vs actual values
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel('Actual Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('Predicted vs Actual Gestational Age')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.text(0.9 * min(y_true), 0.95 * max(y_true), f'RMSE = {rmse:.4f}',
        verticalalignment='top', horizontalalignment='right')
plt.legend()
plt.savefig("./out2/DeepseekR1Distill.py.pdf", format="pdf")