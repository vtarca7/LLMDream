import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_gse_data(dataset_id):
    """Fetch gene expression data from GSE using BECCAM utility."""
    url = f"https://www.ebi.ac.uk/microarray/txedp/{dataset_id}/download?type=BECCAM&format=tsv"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text.split('\n')
        headers = data[0].split()
        data.pop(0)
        df = pd.DataFrame(data, columns=headers)
        return df
    else:
        raise ValueError(f"Failed to fetch GSE{dataset_id}")

def process_data(df):
    """Process the downloaded data and split into train/test sets."""
    # Filter by 'train:ch1' condition
    train_mask = df['train:ch1'] == '1'
    test_mask = df['train:ch1'] == '0'

    # Extract features and target variable
    X_train = df.loc[train_mask, 'Gene Expression'].values.reshape(-1, 1)
    y_train = df.loc[train_mask, 'Gestational Age'].values

    X_test = df.loc[test_mask, 'Gene Expression'].values.reshape(-1, 1)
    y_test = df.loc[test_mask, 'Gestational Age'].values

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and compute RMSE."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, rmse

def plot Results(y_test, y_pred):
    """Plot scatter plot of actual vs predicted values."""
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual Gestational Age')
    plt.ylabel('Predicted Gestational Age')
    plt.title('Actual vs Predicted Gestational Age')
    plt.grid(True)
    plt.savefig("./out2/DeepseekDistill-1.py.pdf", format="pdf")

def main():
    # Fetch the data
    try:
        df = get_gse_data('GSE149440')
        print("Successfully fetched GSE149440 dataset.")

        # Process data
        X_train, y_train, X_test, y_test = process_data(df)

        if len(X_test) == 0 or len(y_test) == 0:
            raise ValueError("No samples found in test set.")

        # Train model
        print("Training model...")
        model = train_model(X_train, y_train)
        print(f"Model trained. Coefficients: {model.coef_}")

        # Evaluate model
        print("\nEvaluating model...")
        y_pred, rmse = evaluate_model(model, X_test, y_test)
        print(f"RMSE on test set: {rmse:.4f}")

        # Plot results
        plot_results(y_test, y_pred)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
