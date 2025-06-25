import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import GEOparse

# Download GEO dataset
geo_id = "GSE149440"
gse = GEOparse.get_GEO(geo=geo_id, destdir=".")

# Extract expression data
gsm_list = gse.gsms.keys()
expression_data = {}
metadata = {}

for gsm_id in gsm_list:
    gsm = gse.gsms[gsm_id]
    expression_data[gsm_id] = gsm.table["VALUE"].values
    metadata[gsm_id] = gsm.metadata

# Convert expression data to DataFrame
expression_df = pd.DataFrame(expression_data)
expression_df = expression_df.T
expression_df.columns = gse.gpls[list(gse.gpls.keys())[0]].table["ID"]

# Extract metadata
meta_df = pd.DataFrame(metadata).T
meta_df["gestational_age"] = meta_df["characteristics_ch1"].apply(lambda x: float([s for s in x if "gestational age:ch1" in s][0].split(": ")[-1]))
meta_df["train"] = meta_df["characteristics_ch1"].apply(lambda x: int([s for s in x if "train:ch1" in s][0].split(": ")[-1]))

# Align expression data with metadata
expression_df = expression_df.loc[meta_df.index]

# Split into training and test sets
train_mask = meta_df["train"] == 1
test_mask = meta_df["train"] == 0

X_train, X_test = expression_df[train_mask], expression_df[test_mask]
y_train, y_test = meta_df.loc[train_mask, "gestational_age"], meta_df.loc[test_mask, "gestational_age"]

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.xlabel("Actual Gestational Age")
plt.ylabel("Predicted Gestational Age")
plt.title("Predicted vs Actual Gestational Age")
plt.savefig("./out2/4o-1.py.pdf", format="pdf")

