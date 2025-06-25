import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load feature data and metadata
df_features_train = pd.read_csv('df_training.csv', index_col=0)
df_metadata_train = pd.read_csv('ano_training.csv')

df_features_test = pd.read_csv('df_test.csv', index_col=0)
df_metadata_test = pd.read_csv('ano_test.csv')

# Merge feature data with metadata
merged_train = df_features_train.merge(df_metadata_train, left_index=True, right_on='Sample_ID')
merged_test = df_features_test.merge(df_metadata_test, left_index=True, right_on='Sample_ID')

# Prepare the training and test datasets
X_train = merged_train.drop(columns=['Sample_ID', 'gestational_age'])
y_train = merged_train['gestational_age']

X_test = merged_test.drop(columns=['Sample_ID', 'gestational_age'])
y_test = merged_test['gestational_age']

# Fit a predictive model (Linear Regression in this case)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Generate scatter plot of predicted vs actual gestational age values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('Predicted vs Actual Gestational Ages')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
plt.text(0.1, 0.9 * max(y_test.max(), y_pred.max()), f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes)
plt.grid(True)
plt.savefig("./out2/phi4-3.py.pdf", format="pdf")
