import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load and reformat feature data
# -------------------------------

# Read training and test feature files.
# Note: The first column contains the feature names; the remaining columns are sample IDs.
df_train = pd.read_csv('df_training.csv')
df_test = pd.read_csv('df_test.csv')

# Set the first column as the index (feature names) and then transpose so that rows are samples.
df_train = df_train.set_index(df_train.columns[0]).T
df_test = df_test.set_index(df_test.columns[0]).T

# -------------------------------
# 2. Load metadata and merge with features
# -------------------------------

# Read metadata files; these include Sample_ID and GA.
meta_train = pd.read_csv('ano_training.csv')
meta_test = pd.read_csv('ano_test.csv')

# Merge the feature data with metadata on Sample_ID.
# After transposition, the index of df_train/df_test is the Sample_ID.
train_data = df_train.merge(meta_train[['Sample_ID', 'GA']], left_index=True, right_on='Sample_ID')
test_data = df_test.merge(meta_test[['Sample_ID', 'GA']], left_index=True, right_on='Sample_ID')

# Separate features (X) from the target (y). Do not include any other metadata variables.
X_train = train_data.drop(columns=['Sample_ID', 'GA'])
y_train = train_data['GA']

X_test = test_data.drop(columns=['Sample_ID', 'GA'])
y_test = test_data['GA']

# -------------------------------
# 3. Fit the predictive model
# -------------------------------

# We use a pipeline that first scales the features and then fits a Lasso model with 5-fold CV.
# Lasso is chosen here for its built-in feature selection (important for high-dimensional data).
model = make_pipeline(StandardScaler(), LassoCV(cv=5, n_jobs=-1, random_state=42))

# Fit the model on the training data.
model.fit(X_train, y_train)

# -------------------------------
# 4. Evaluate the model on the test set
# -------------------------------

# Generate predictions on the test set.
y_pred = model.predict(X_test)

# Compute the Root Mean Squared Error (RMSE).
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse)

# -------------------------------
# 5. Generate a scatter plot of predicted vs actual GA values
# -------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label=f'RMSE = {rmse:.2f}')
plt.xlabel('Actual Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('Predicted vs Actual Gestational Age')
plt.legend()

# Optionally, add a reference line (ideal predictions would fall on this line).
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.savefig("./out2/o3-high-3.py.pdf",format="pdf")
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
df.to_csv("./out2/o3-high-3.py.csv", index=False)
