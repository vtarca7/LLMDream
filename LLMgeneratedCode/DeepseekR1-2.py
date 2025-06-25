import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_and_process(feature_path, metadata_path, collect_wk_threshold):
    # Load and merge data
    features = pd.read_csv(feature_path)
    metadata = pd.read_csv(metadata_path)
    merged = pd.merge(features, metadata, on='specimen')
    
    # Filter and select latest specimen per participant
    filtered = merged[merged['collect_wk'] < collect_wk_threshold]
    selected = filtered.groupby('participant_id').apply(
        lambda x: x.nlargest(1, 'collect_wk')).reset_index(drop=True)
    return selected

def prepare_data(selected_data, outcome_threshold, train_features_columns):
    # Create outcome variable
    selected_data['outcome'] = (selected_data['delivery_wk'] < outcome_threshold).astype(int)
    # Select common species features
    return selected_data[train_features_columns], selected_data['outcome']

# ==========================
# Analysis for PTB (delivery_wk < 37)
# ==========================
# Process training data
train_ptb = load_and_process('training_species_abundance.csv', 
                            'training_metadata.csv', 32)
# Process validation data
val_ptb = load_and_process('validation_species_abundance.csv', 
                          'validation_metadata.csv', 32)

# Get common species features
common_species = list(set(train_ptb.columns) & set(val_ptb.columns) - 
                {'specimen', 'participant_id', 'collect_wk', 'delivery_wk'})

# Prepare data
X_train_ptb, y_train_ptb = prepare_data(train_ptb, 37, common_species)
X_val_ptb, y_val_ptb = prepare_data(val_ptb, 37, common_species)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ptb)
X_val_scaled = scaler.transform(X_val_ptb)

# Train model
model = LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs', random_state=42)
model.fit(X_train_scaled, y_train_ptb)

# Evaluate
y_pred = model.predict_proba(X_val_scaled)[:, 1]
auc = roc_auc_score(y_val_ptb, y_pred)
print(f"PTB AUC: {auc:.3f}")

# Plot ROC
fpr, tpr, _ = roc_curve(y_val_ptb, y_pred)
plt.figure()
plt.plot(fpr, tpr, label=f'PTB (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for PTB Prediction')
plt.legend()
plt.savefig("./out2/DeepseekR1-2_A.py.pdf",format="pdf")
# ==========================
# Analysis for EarlyPTB (delivery_wk < 32)
# ==========================
# Process training data
train_early = load_and_process('training_species_abundance.csv', 
                              'training_metadata.csv', 28)
# Process validation data
val_early = load_and_process('validation_species_abundance.csv', 
                            'validation_metadata.csv', 28)

# Get common species features
common_species_early = list(set(train_early.columns) & set(val_early.columns) - 
                      {'specimen', 'participant_id', 'collect_wk', 'delivery_wk'})

# Prepare data
X_train_early, y_train_early = prepare_data(train_early, 32, common_species_early)
X_val_early, y_val_early = prepare_data(val_early, 32, common_species_early)

# Scale features
scaler_early = StandardScaler()
X_train_scaled_early = scaler_early.fit_transform(X_train_early)
X_val_scaled_early = scaler_early.transform(X_val_early)

# Train model
model_early = LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs', random_state=42)
model_early.fit(X_train_scaled_early, y_train_early)

# Evaluate
y_pred_early = model_early.predict_proba(X_val_scaled_early)[:, 1]
auc_early = roc_auc_score(y_val_early, y_pred_early)
print(f"EarlyPTB AUC: {auc_early:.3f}")

# Plot ROC
fpr_e, tpr_e, _ = roc_curve(y_val_early, y_pred_early)
plt.figure()
plt.plot(fpr_e, tpr_e, label=f'EarlyPTB (AUC = {auc_early:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for EarlyPTB Prediction')
plt.legend()
plt.savefig("./out2/DeepseekR1-2_B.py.pdf",format="pdf")
