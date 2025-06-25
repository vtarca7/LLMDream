import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def analyze_outcome(outcome_type, collect_wk_threshold, delivery_wk_threshold, collect_wk_cutoff):
    """
    Analyzes the prediction of a binary outcome (PTB or EarlyPTB) using microbiome features.

    Args:
        outcome_type (str): "PTB" or "EarlyPTB", the type of preterm birth to predict.
        collect_wk_threshold (int): collect_wk threshold for selecting samples (e.g., 32 for PTB, 28 for EarlyPTB).
        delivery_wk_threshold (int): delivery_wk threshold defining the positive outcome (e.g., 37 for PTB, 32 for EarlyPTB).
        collect_wk_cutoff (int): collect_wk upper bound for filtering initial data.

    Returns:
        None: Prints AUC-ROC score and plots ROC curve for the validation set.
    """

    print(f"\nAnalyzing outcome: {outcome_type}")

    # Load data
    train_feature_df = pd.read_csv('training_species_abundance.csv', index_col=0)
    val_feature_df = pd.read_csv('validation_species_abundance.csv', index_col=0)
    train_metadata_df = pd.read_csv('training_metadata.csv')
    val_metadata_df = pd.read_csv('validation_metadata.csv')

    # Identify common features
    common_features = list(set(train_feature_df.columns) & set(val_feature_df.columns))
    train_feature_df = train_feature_df[common_features]
    val_feature_df = val_feature_df[common_features]

    # --- Process Training Data ---
    train_metadata_filtered = train_metadata_df[train_metadata_df['collect_wk'] < collect_wk_cutoff].copy() # Initial filter before max collect_wk selection

    # Sort by collect_wk to easily pick the last entry
    train_metadata_filtered = train_metadata_filtered.sort_values(by='collect_wk')

    # Get the last entry with collect_wk < collect_wk_threshold for each participant
    train_metadata_last_entries = train_metadata_filtered[train_metadata_filtered['collect_wk'] < collect_wk_threshold].groupby('participant_id').last().reset_index()


    # Merge with feature data
    train_merged_df = pd.merge(train_metadata_last_entries, train_feature_df.reset_index(), on='specimen', how='inner')

    if train_merged_df.empty:
        print(f"No training data available after filtering for {outcome_type}. Check collect_wk thresholds.")
        return


    # Define PTB outcome
    train_merged_df[outcome_type] = (train_merged_df['delivery_wk'] < delivery_wk_threshold).astype(int)


    # --- Process Validation Data ---
    val_metadata_filtered = val_metadata_df[val_metadata_df['collect_wk'] < collect_wk_cutoff].copy() # Initial filter before max collect_wk selection
    val_metadata_filtered = val_metadata_filtered.sort_values(by='collect_wk')
    val_metadata_last_entries = val_metadata_filtered[val_metadata_filtered['collect_wk'] < collect_wk_threshold].groupby('participant_id').last().reset_index()


    # Merge with feature data
    val_merged_df = pd.merge(val_metadata_last_entries, val_feature_df.reset_index(), on='specimen', how='inner')

    if val_merged_df.empty:
        print(f"No validation data available after filtering for {outcome_type}. Check collect_wk thresholds.")
        return

    # Define PTB outcome
    val_merged_df[outcome_type] = (val_merged_df['delivery_wk'] < delivery_wk_threshold).astype(int)


    # Prepare data for modeling
    features = common_features
    X_train = train_merged_df[features]
    y_train = train_merged_df[outcome_type]
    X_val = val_merged_df[features]
    y_val = val_merged_df[outcome_type]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced') # Use class_weight to handle potential imbalance
    model.fit(X_train_scaled, y_train)

    # Make predictions on validation set
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    auc_roc = roc_auc_score(y_val, y_pred_proba)
    print(f"AUC-ROC on Validation Set for {outcome_type}: {auc_roc:.4f}")

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {outcome_type} Prediction')
    plt.legend(loc="lower right")
    plt.savefig("./out2/gemini-2.py_"+outcome_type+".pdf", format="pdf")
    df = pd.DataFrame({'y_test': y_val, 'y_prob': y_pred_proba})
    df.to_csv("./out2/gemini-2.py_"+outcome_type+".csv", index=False)


# --- Run analysis for PTB ---
analyze_outcome(outcome_type='PTB', collect_wk_threshold=32, delivery_wk_threshold=37, collect_wk_cutoff=32)

# --- Run analysis for EarlyPTB ---
analyze_outcome(outcome_type='EarlyPTB', collect_wk_threshold=28, delivery_wk_threshold=32, collect_wk_cutoff=28)
