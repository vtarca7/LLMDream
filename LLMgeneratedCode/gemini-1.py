import GEOparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Download the dataset
geo_id = "GSE149440"
dest_dir = "./data/"
gse = GEOparse.get_GEO(geo=geo_id, destdir=dest_dir)

# Access expression data and metadata
expression_data_df = gse.pivot_samples('VALUE')
metadata_df = gse.phenotype_data

# Convert gestational age to numeric and handle potential errors
metadata_df['gestational age:ch1'] = pd.to_numeric(metadata_df['gestational age:ch1'], errors='coerce')

# Identify training and test samples
train_samples_metadata = metadata_df[metadata_df['train:ch1'] == '1']
test_samples_metadata = metadata_df[metadata_df['train:ch1'] == '0']

# Get sample names for train and test sets
train_sample_names = train_samples_metadata.index
test_sample_names = test_samples_metadata.index

# Filter expression data for train and test samples
train_expression_data = expression_data_df[train_sample_names]
test_expression_data = expression_data_df[test_sample_names]

# Align metadata and expression data and drop samples with NaN gestational age in train set
train_metadata_aligned = train_samples_metadata.loc[train_sample_names].dropna(subset=['gestational age:ch1'])
train_expression_data_aligned = train_expression_data[train_metadata_aligned.index]

# Prepare training data
X_train = train_expression_data_aligned.T # Transpose so genes are features and samples are rows
y_train = train_metadata_aligned['gestational age:ch1']

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prepare test data, aligning and handling potential NaN gestational age in test set (though not strictly needed for prediction, good for evaluation if present)
test_metadata_aligned = test_samples_metadata.loc[test_sample_names].dropna(subset=['gestational age:ch1'])
test_expression_data_aligned = test_expression_data[test_metadata_aligned.index] # Use aligned indices

X_test = test_expression_data_aligned.T # Transpose for test data as well
y_test_actual = test_metadata_aligned['gestational age:ch1']

# Predict gestational age for test set
y_test_predicted = model.predict(X_test)

# Evaluate the model: Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_predicted))
print(f"Root Mean Squared Error (RMSE) on the test set: {rmse:.4f}")

# Generate scatter plot of predicted vs actual gestational age
plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, y_test_predicted, alpha=0.6)
plt.xlabel("Actual Gestational Age")
plt.ylabel("Predicted Gestational Age")
plt.title("Predicted vs Actual Gestational Age (Test Set)")
plt.plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], color='red', linestyle='--', linewidth=1) # Diagonal line for perfect prediction
plt.grid(True)
plt.savefig("gestational_age_prediction_scatter_plot.png") # Save the plot
plt.savefig("./out2/gemini-1.py.pdf", format="pdf")