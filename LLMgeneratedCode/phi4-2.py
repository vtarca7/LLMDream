import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load datasets
training_species = pd.read_csv('training_species_abundance.csv')
validation_species = pd.read_csv('validation_species_abundance.csv')

training_metadata = pd.read_csv('training_metadata.csv')
validation_metadata = pd.read_csv('validation_metadata.csv')

# Merge species and metadata on 'specimen'
train_data = training_species.merge(training_metadata, on='specimen')
valid_data = validation_species.merge(validation_metadata, on='specimen')

# Filter for highest collect_wk < 32
def filter_latest_samples(data):
    filtered = data[data['collect_wk'] < 32].groupby('participant_id').apply(
        lambda x: x.loc[x['collect_wk'].idxmax()]
    ).reset_index(drop=True)
    return filtered

train_data_filtered = filter_latest_samples(train_data)

# Filter for PTB outcome
train_data_ptb = train_data_filtered[train_data_filtered['delivery_wk'] < 37]

# Get common features between training and validation sets
common_features = set(train_data_ptb.columns) & set(valid_data.columns)
features = [col for col in common_features if 'participant_id' not in col and 'specimen' not in col]

# Prepare training data
X_train = train_data_ptb[features]
y_train = (train_data_ptb['delivery_wk'] < 37).astype(int)

# Filter validation set for highest collect_wk < 32
valid_data_filtered = filter_latest_samples(valid_data)
X_valid = valid_data_filtered[features]
y_valid = (valid_data_filtered['delivery_wk'] < 37).astype(int)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Predict and evaluate on validation set
y_pred_prob = model.predict_proba(X_valid)[:, 1]
auc_roc_ptb = roc_auc_score(y_valid, y_pred_prob)
print(f'PTB AUC ROC: {auc_roc_ptb}')

# Plot ROC curve for PTB
fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_roc_ptb:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for PTB Prediction')
plt.legend(loc='lower right')
plt.savefig("./out2/phi4-2.py_A.pdf", format="pdf")

# Repeat analysis for EarlyPTB outcome
train_data_earlyptb = train_data_filtered[train_data_filtered['collect_wk'] < 28]
valid_data_earlyptb = valid_data_filtered[valid_data_filtered['collect_wk'] < 28]

X_train_earlyptb = train_data_earlyptb[features]
y_train_earlyptb = (train_data_earlyptb['delivery_wk'] < 32).astype(int)

X_valid_earlyptb = valid_data_earlyptb[features]
y_valid_earlyptb = (valid_data_earlyptb['delivery_wk'] < 32).astype(int)

# Train a logistic regression model for EarlyPTB
model_earlyptb = LogisticRegression(max_iter=1000, solver='liblinear')
model_earlyptb.fit(X_train_earlyptb, y_train_earlyptb)

# Predict and evaluate on validation set for EarlyPTB
y_pred_prob_earlyptb = model_earlyptb.predict_proba(X_valid_earlyptb)[:, 1]
auc_roc_earlyptb = roc_auc_score(y_valid_earlyptb, y_pred_prob_earlyptb)
print(f'EarlyPTB AUC ROC: {auc_roc_earlyptb}')

# Plot ROC curve for EarlyPTB
fpr_earlyptb, tpr_earlyptb, _ = roc_curve(y_valid_earlyptb, y_pred_prob_earlyptb)
plt.figure()
plt.plot(fpr_earlyptb, tpr_earlyptb, label=f'AUC = {auc_roc_earlyptb:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for EarlyPTB Prediction')
plt.legend(loc='lower right')
plt.savefig("./out2/phi4-2.py_B.pdf", format="pdf")
