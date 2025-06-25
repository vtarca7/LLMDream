import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Download the dataset
# You can download the dataset from GEO (Gene Expression Omnibus) and save it locally.
# For this example, let's assume the files are already downloaded and saved.

# Load the metadata file
metadata = pd.read_csv('GSE149440_metadata.txt', delimiter='\t')

# Load the expression data file
expression_data = pd.read_csv('GSE149440_expression_data.csv', index_col=0)

# Step 2: Preprocess the data
# Filter out samples that are not assigned to the training set
train_samples = metadata[metadata['train:ch1'] == '1']
test_samples = metadata[metadata['train:ch1'] == '0']

# Select only the relevant columns for training and testing
X_train = expression_data.loc[train_samples.index]
y_train = train_samples['gestational age:ch1'].astype(float)

X_test = expression_data.loc[test_samples.index]
y_test = test_samples['gestational age:ch1'].astype(float)

# Step 3: Fit the prediction model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error (RMSE) on the test set: {rmse}')

# Step 5: Generate a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('Scatter Plot of Predicted vs Actual Gestational Age')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
plt.savefig("./out2/qwen-1.py.pdf", format="pdf")