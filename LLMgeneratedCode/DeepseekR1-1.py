import GEOparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Download and parse the GEO dataset
gse = GEOparse.get_GEO(geo="GSE149440")

# Initialize lists to hold sample IDs and target values
train_samples, test_samples = [], []
y_train, y_test = [], []

# Iterate through each sample in the dataset
for gsm in gse.gsms.values():
    # Extract training flag and gestational age from metadata
    train_flag = gsm.metadata.get('train:ch1', [''])[0]
    ga = gsm.metadata.get('gestational age:ch1', [''])[0]
    
    if not train_flag or not ga:
        continue  # Skip samples with missing data
    
    try:
        ga_value = float(ga)
    except ValueError:
        continue  # Skip invalid gestational age values
    
    if train_flag == '1':
        train_samples.append(gsm.name)
        y_train.append(ga_value)
    elif train_flag == '0':
        test_samples.append(gsm.name)
        y_test.append(ga_value)

# Extract expression data and transpose to samples x genes format
X_train = gse.expression_data[train_samples].T
X_test = gse.expression_data[test_samples].T

# Convert target lists to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Train Ridge regression model
model = Ridge()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate and print RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f}")

# Generate scatter plot of predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('Actual vs Predicted Gestational Age on Test Set')
plt.grid(True)
plt.savefig("./out2/DeepseekR1-1.py.pdf", format="pdf")
