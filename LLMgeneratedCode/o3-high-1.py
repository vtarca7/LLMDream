import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def main():
    # ---------------------------
    # 1. Download the GEO dataset
    # ---------------------------
    print("Downloading GEO dataset GSE149440 ...")
    gse = GEOparse.get_GEO("GSE149440", destdir="./")

    # ------------------------------------------------
    # 2. Extract the expression data (gene expression)
    # ------------------------------------------------
    # GEOparse can pivot the data so that rows=genes and columns=samples.
    # We then transpose to have samples as rows (each with tens of thousands of gene features).
    expr_df = gse.pivot_and_annotate('VALUE')
    X = expr_df.T

    # Optionally, if there are any non-numeric annotation columns in X, keep only numeric ones.
    X = X.select_dtypes(include=[np.number])

    # ----------------------------------------
    # 3. Build a metadata DataFrame from GSMs
    # ----------------------------------------
    # For each sample (GSM), extract the metadata entries:
    #   - "train:ch1" (used to split training/test)
    #   - "gestational age:ch1" (target variable)
    metadata = {}
    for gsm_name, gsm in gse.gsms.items():
        md = gsm.metadata
        train_val = md.get("train:ch1", [None])[0]
        ga_val = md.get("gestational age:ch1", [None])[0]
        metadata[gsm_name] = {"train": train_val, "gestational_age": ga_val}
    metadata_df = pd.DataFrame.from_dict(metadata, orient='index')

    # Ensure the samples in the expression matrix and metadata match
    common_samples = X.index.intersection(metadata_df.index)
    X = X.loc[common_samples]
    metadata_df = metadata_df.loc[common_samples]

    # Convert gestational age to numeric (if it isn’t already)
    metadata_df['gestational_age'] = pd.to_numeric(metadata_df['gestational_age'], errors='coerce')

    # Drop any samples with missing gestational age or train flag
    metadata_df = metadata_df.dropna(subset=['gestational_age', 'train'])
    X = X.loc[metadata_df.index]

    # ------------------------------------------------
    # 4. Split the data into training and test sets
    # ------------------------------------------------
    # Training set: samples where train:ch1 == '1'
    # Test set: samples where train:ch1 == '0'
    train_samples = metadata_df[metadata_df['train'] == '1'].index
    test_samples  = metadata_df[metadata_df['train'] == '0'].index

    X_train = X.loc[train_samples].astype(float)
    y_train = metadata_df.loc[train_samples, 'gestational_age']

    X_test = X.loc[test_samples].astype(float)
    y_test = metadata_df.loc[test_samples, 'gestational_age']

    # ------------------------------------------------
    # 5. Train a prediction model (Ridge Regression)
    # ------------------------------------------------
    model = Ridge(alpha=1.0)
    print("Fitting the regression model on the training set ...")
    model.fit(X_train, y_train)

    # Predict gestational age on the test set
    y_pred = model.predict(X_test)

    # --------------------------------------
    # 6. Evaluate the model (compute RMSE)
    # --------------------------------------
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Test RMSE: {:.2f}".format(rmse))

    # --------------------------------------------------------
    # 7. Generate a scatter plot: Predicted vs Actual values
    # --------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
    plt.xlabel("Actual Gestational Age")
    plt.ylabel("Predicted Gestational Age")
    plt.title("Predicted vs Actual Gestational Age (Test Set)")
    # Plot the diagonal line for reference
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.tight_layout()
    plt.savefig("./out2/o3-high-1.py.pdf", format="pdf")

if __name__ == "__main__":
    main()
