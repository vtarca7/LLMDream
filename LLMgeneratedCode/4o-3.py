import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load training data
feature_train = pd.read_csv("df_training.csv")
metadata_train = pd.read_csv("ano_training.csv")

# Load test data
feature_test = pd.read_csv("df_test.csv")
metadata_test = pd.read_csv("ano_test.csv")

# Transpose feature data so samples are rows and features are columns
feature_train = feature_train.set_index(feature_train.columns[0]).T
feature_test = feature_test.set_index(feature_test.columns[0]).T

# Ensure the index (Sample_ID) matches between features and metadata
metadata_train = metadata_train.set_index("Sample_ID").loc[feature_train.index]
metadata_test = metadata_test.set_index("Sample_ID").loc[feature_test.index]

# Extract gestational age from metadata
y_train = metadata_train["GA"].values
y_test = metadata_test["GA"].values

# Convert feature data to numpy arrays
X_train = feature_train.values
X_test = feature_test.values

# Train a ridge regression model (L2 regularization for high-dimensional data)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f}")

# Scatter plot of actual vs predicted GA
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label=f"RMSE: {rmse:.2f}")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal Fit")
plt.xlabel("Actual GA")
plt.ylabel("Predicted GA")
plt.legend()
plt.title("Predicted vs Actual Gestational Age")
plt.savefig("./out2/4o-3.py.pdf", format="pdf")
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
df.to_csv("./out2/4o-3.py.csv", index=False)
