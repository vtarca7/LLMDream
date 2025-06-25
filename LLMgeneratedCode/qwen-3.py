import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
df_training = pd.read_csv('df_training.csv', index_col=0)
ano_training = pd.read_csv('ano_training.csv')
df_test = pd.read_csv('df_test.csv', index_col=0)
ano_test = pd.read_csv('ano_test.csv')

# Merge feature data with metadata using Sample_ID
merged_training = df_training.T.merge(ano_training, left_index=True, right_on='Sample_ID')
merged_test = df_test.T.merge(ano_test, left_index=True, right_on='Sample_ID')

# Split into features and target variable
X_train = merged_training.drop(columns=['gestational_age'])
y_train = merged_training['gestational_age']
X_test = merged_test.drop(columns=['gestational_age'])
y_test = merged_test['gestational_age']

# Preprocess the data (e.g., scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a predictive model (Random Forest Regressor in this case)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Generate a scatter plot of predicted vs actual gestational age
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('Scatter Plot of Predicted vs Actual Gestational Age')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.legend(['RMSE = {:.2f}'.format(rmse)])
plt.savefig("./out2/qwen-3.py.pdf", format="pdf")
