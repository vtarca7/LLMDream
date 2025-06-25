import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load feature data and transpose to samples x features format
def load_features(path):
    df = pd.read_csv(path, index_col=0)
    return df.T  # Transpose features to samples x features

X_train = load_features('df_training.csv')
X_test = load_features('df_test.csv')

# Load metadata and align with feature data
def load_metadata(path, sample_ids):
    ano = pd.read_csv(path).set_index('Sample_ID')
    return ano.loc[sample_ids, 'GA']  # Align using sample_ids

y_train = load_metadata('ano_training.csv', X_train.index)
y_test = load_metadata('ano_test.csv', X_test.index)

# Create pipeline with standardization and Ridge regression
# Ridge handles high-dimensional data with L2 regularization
model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0, solver='sag')  # solver='sag' handles large feature space
)

# Train model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE: {rmse:.4f}')

# Generate scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label=f'RMSE: {rmse:.2f}')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('Actual vs Predicted Gestational Age (Test Set)')
plt.legend()
plt.savefig("./out2/DeepseekR1-3.py.pdf",format="pdf")
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
df.to_csv("./out2/DeepseekR1-3.py.csv", index=False)