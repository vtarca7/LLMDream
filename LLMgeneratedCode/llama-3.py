# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data
df_training = pd.read_csv('df_training.csv')
df_test = pd.read_csv('df_test.csv')

ano_training = pd.read_csv('ano_training.csv', index_col='Sample_ID')
ano_test = pd.read_csv('ano_test.csv', index_col='Sample_ID')

# Merge data with metadata
df_training['gestational_age'] = df_training['Sample_ID'].map(ano_training['gestational_age'])
df_test['gestational_age'] = df_test['Sample_ID'].map(ano_test['gestational_age'])

# Split data into features (X) and target variable (y)
X_train = df_training.drop(['gestational_age'], axis=1)
y_train = df_training['gestational_age']
X_test = df_test.drop(['gestational_age'], axis=1)
y_test = df_test['gestational_age']

# Train a random forest regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")

# Generate a scatter plot of predicted vs actual values for the test set
plt.figure(figsize=(10,6))
sns.scatterplot(x=predictions, y=y_test)
plt.axhline(y=0, color='r', linestyle='--', label='y = 0')
plt.legend()
plt.title("Predicted vs Actual Gestational Age")
plt.xlabel("Predicted Gestational Age")
plt.ylabel("Actual Gestational Age")
plt.savefig("./out2/llama-3.py.pdf", format="pdf")
# Plot RMSE in the scatter plot legend
plt.legend([f"RMSE: {rmse}"])
