import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

# Load training and validation data
train_features = pd.read_csv('training_species_abundance.csv')
val_features = pd.read_csv('validation_species_abundance.csv')

# Discard features not in common between training and validation sets
common_features = set(train_features.columns) & set(val_features.columns)
train_features = train_features[common_features]
val_features = val_features[common_features]

# Load metadata for training and validation sets
train_metadata = pd.read_csv('training_metadata.csv')
val_metadata = pd.read_csv('validation_metadata.csv')

# Merge metadata with features for each participant_id
train_data = train_features.merge(train_metadata[['participant_id', 'delivery_wk']], on='specimen')
val_data = val_features.merge(val_metadata[['participant_id', 'delivery_wk']], on='specimen')

# Select only the entry with highest collect_wk value < 32 for each participant_id in both datasets
train_data = train_data[(train_data['collect_wk'] < 32) & (train_data.groupby('participant_id')['collect_wk'].transform('max') == train_data['collect_wk'])]
val_data = val_data[(val_data['collect_wk'] < 32) & (val_data.groupby('participant_id')['collect_wk'].transform('max') == val_data['collect_wk'])]

# Fit and evaluate random forest model for EarlyPTB outcome
early_ptb_model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['participant_id', 'delivery_wk'], axis=1), train_data['delivery_wk'], test_size=0.2, random_state=42)

early_ptb_model.fit(X_train, y_train)
y_pred_proba = early_ptb_model.predict_proba(X_test)[:, 1]
auc_early_ptb = roc_auc_score(y_test, y_pred_proba)
print('AUC ROC value for EarlyPTB outcome:', auc_early_ptb)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (EarlyPTB)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("./out2/llama-2.py_A.pdf", format="pdf")

# Repeat analysis for PTB outcome defined as delivery_wk < 37
ptb_model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['participant_id', 'delivery_wk'], axis=1), train_data['delivery_wk'], test_size=0.2, random_state=42)

ptb_model.fit(X_train, y_train)
y_pred_proba = ptb_model.predict_proba(X_test)[:, 1]
auc_ptb = roc_auc_score(y_test, y_pred_proba)
print('AUC ROC value for PTB outcome:', auc_ptb)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (PTB)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("./out2/llama-2.py_B.pdf", format="pdf")
