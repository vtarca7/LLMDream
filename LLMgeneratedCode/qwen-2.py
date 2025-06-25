import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load data
training_species_abundance = pd.read_csv('training_species_abundance.csv')
validation_species_abundance = pd.read_csv('validation_species_abundance.csv')
training_metadata = pd.read_csv('training_metadata.csv')
validation_metadata = pd.read_csv('validation_metadata.csv')

# Merge feature and metadata for training set
train_df = training_species_abundance.merge(training_metadata, on='specimen')

# Select the last specimen with collect_wk < 32 for each participant_id
train_last_specimen = train_df[train_df['collect_wk'] < 32].drop_duplicates('participant_id', keep='last')
train_features = train_last_specimen.drop(['specimen', 'participant_id', 'collect_wk'], axis=1)
train_labels = (train_last_specimen['delivery_wk'] < 37).astype(int)

# Merge feature and metadata for validation set
val_df = validation_species_abundance.merge(validation_metadata, on='specimen')

# Select the last specimen with collect_wk < 28 for each participant_id for PTB prediction
val_last_specimen_PTB = val_df[val_df['collect_wk'] < 32].drop_duplicates('participant_id', keep='last')
val_features_PTB = val_last_specimen_PTB.drop(['specimen', 'participant_id', 'collect_wk'], axis=1)
val_labels_PTB = (val_last_specimen_PTB['delivery_wk'] < 37).astype(int)

# Select the last specimen with collect_wk < 28 for each participant_id for EarlyPTB prediction
val_last_specimen_EarlyPTB = val_df[val_df['collect_wk'] < 28].drop_duplicates('participant_id', keep='last')
val_features_EarlyPTB = val_last_specimen_EarlyPTB.drop(['specimen', 'participant_id', 'collect_wk'], axis=1)
val_labels_EarlyPTB = (val_last_specimen_EarlyPTB['delivery_wk'] < 32).astype(int)

# Discard features not in common between training and validation sets
common_features = train_features.columns.intersection(val_features_PTB.columns).intersection(val_features_EarlyPTB.columns)
train_features = train_features[common_features]
val_features_PTB = val_features_PTB[common_features]
val_features_EarlyPTB = val_features_EarlyPTB[common_features]

# Standardize the features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_PTB_scaled = scaler.transform(val_features_PTB)
val_features_EarlyPTB_scaled = scaler.transform(val_features_EarlyPTB)

# Fit a Random Forest model for PTB prediction
model_PTB = RandomForestClassifier(n_estimators=100, random_state=42)
model_PTB.fit(train_features_scaled, train_labels)

# Predict probabilities on the validation set
val_probabilities_PTB = model_PTB.predict_proba(val_features_PTB_scaled)[:, 1]

# Calculate AUC ROC for PTB
auc_roc_PTB = roc_auc_score(val_labels_PTB, val_probabilities_PTB)
print(f'AUC ROC for PTB prediction: {auc_roc_PTB}')

# Plot the ROC curve for PTB
fpr_PTB, tpr_PTB, thresholds_PTB = roc_curve(val_labels_PTB, val_probabilities_PTB)
roc_auc_PTB = auc(fpr_PTB, tpr_PTB)

plt.figure()
plt.plot(fpr_PTB, tpr_PTB, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_PTB:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for PTB Prediction')
plt.legend(loc="lower right")
plt.savefig("./out2/qwen-2.py_A.pdf", format="pdf")

# Fit a Random Forest model for EarlyPTB prediction
model_EarlyPTB = RandomForestClassifier(n_estimators=100, random_state=42)
model_EarlyPTB.fit(train_features_scaled, train_labels)

# Predict probabilities on the validation set
val_probabilities_EarlyPTB = model_EarlyPTB.predict_proba(val_features_EarlyPTB_scaled)[:, 1]

# Calculate AUC ROC for EarlyPTB
auc_roc_EarlyPTB = roc_auc_score(val_labels_EarlyPTB, val_probabilities_EarlyPTB)
print(f'AUC ROC for EarlyPTB prediction: {auc_roc_EarlyPTB}')

# Plot the ROC curve for EarlyPTB
fpr_EarlyPTB, tpr_EarlyPTB, thresholds_EarlyPTB = roc_curve(val_labels_EarlyPTB, val_probabilities_EarlyPTB)
roc_auc_EarlyPTB = auc(fpr_EarlyPTB, tpr_EarlyPTB)

plt.figure()
plt.plot(fpr_EarlyPTB, tpr_EarlyPTB, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_EarlyPTB:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for EarlyPTB Prediction')
plt.legend(loc="lower right")
plt.savefig("./out2/qwen-2.py_B.pdf", format="pdf")
