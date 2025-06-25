import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Load microbiome feature data
train_features = pd.read_csv("training_species_abundance.csv")
valid_features = pd.read_csv("validation_species_abundance.csv")

# Load metadata
train_metadata = pd.read_csv("training_metadata.csv")
valid_metadata = pd.read_csv("validation_metadata.csv")

# Ensure only common features are used
common_features = set(train_features.columns).intersection(set(valid_features.columns))
common_features.discard("specimen")  # Remove the linking column
train_features = train_features[["specimen"] + list(common_features)]
valid_features = valid_features[["specimen"] + list(common_features)]

# Merge metadata with feature data
train_data = train_metadata.merge(train_features, on="specimen")
valid_data = valid_metadata.merge(valid_features, on="specimen")

def preprocess_data(metadata, features, collect_wk_threshold, delivery_wk_threshold):
    """Filters and prepares the dataset for modeling."""
    
    # Define PTB outcome
    metadata["PTB"] = (metadata["delivery_wk"] < delivery_wk_threshold).astype(int)
    
    # Select the latest specimen before the threshold for each participant
    metadata = metadata[metadata["collect_wk"] < collect_wk_threshold]
    metadata = metadata.sort_values(by=["participant_id", "collect_wk"], ascending=[True, False])
    metadata = metadata.drop_duplicates(subset=["participant_id"], keep="first")
    
    # Merge with features
    final_data = metadata.merge(features, on="specimen")
    
    # Prepare feature matrix and target variable
    X = final_data.drop(columns=["specimen", "participant_id", "delivery_wk", "PTB"])
    y = final_data["PTB"]
    
    return X, y

# Process data for PTB prediction
X_train, y_train = preprocess_data(train_metadata, train_features, collect_wk_threshold=32, delivery_wk_threshold=37)
X_valid, y_valid = preprocess_data(valid_metadata, valid_features, collect_wk_threshold=32, delivery_wk_threshold=37)

# Train a RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred_prob = clf.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, y_pred_prob)
print(f"AUC-ROC for PTB Prediction: {auc_score:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'PTB (AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for PTB Prediction')
plt.legend()
plt.savefig("./out2/4o-2.py_A.pdf", format="pdf")

# Repeat for EarlyPTB (delivery_wk < 32, collect_wk < 28)
X_train, y_train = preprocess_data(train_metadata, train_features, collect_wk_threshold=28, delivery_wk_threshold=32)
X_valid, y_valid = preprocess_data(valid_metadata, valid_features, collect_wk_threshold=28, delivery_wk_threshold=32)

clf.fit(X_train, y_train)
y_pred_prob = clf.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, y_pred_prob)
print(f"AUC-ROC for EarlyPTB Prediction: {auc_score:.4f}")

# Plot ROC Curve for EarlyPTB
fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'EarlyPTB (AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for EarlyPTB Prediction')
plt.legend()
plt.savefig("./out2/4o-2.py_B.pdf", format="pdf")
