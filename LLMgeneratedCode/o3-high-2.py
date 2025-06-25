import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# =============================================================================
# Helper functions
# =============================================================================
def load_and_process(species_file, metadata_file, collect_wk_threshold, delivery_wk_threshold, outcome_name):
    """
    Reads in species and metadata files, merges them by specimen,
    filters to specimens with collect_wk below a threshold, selects per-participant 
    the specimen with the highest collect_wk (i.e. the last available specimen within that window),
    and creates a binary outcome column (named outcome_name) defined by delivery_wk threshold.
    
    Parameters:
      species_file: Path to the species abundance CSV (which includes a 'specimen' column)
      metadata_file: Path to the metadata CSV (which includes 'specimen', 'participant_id', 'collect_wk', and 'delivery_wk')
      collect_wk_threshold: Only specimens with collect_wk < this value are considered.
      delivery_wk_threshold: Outcome is defined as delivery_wk < this threshold.
      outcome_name: Name to give to the binary outcome column.
      
    Returns:
      A processed pandas DataFrame.
    """
    # Read in data
    species = pd.read_csv(species_file)
    metadata = pd.read_csv(metadata_file)
    
    # Merge on specimen
    df = pd.merge(metadata, species, on='specimen', how='inner')
    
    # Keep only specimens with collect_wk below the threshold
    df = df[df['collect_wk'] < collect_wk_threshold].copy()
    
    # For each participant, select the row with the highest collect_wk
    df = df.sort_values('collect_wk').groupby('participant_id', as_index=False).last()
    
    # Create the binary outcome column: 1 if delivery_wk is below the given threshold, else 0.
    df[outcome_name] = (df['delivery_wk'] < delivery_wk_threshold).astype(int)
    
    return df

def get_common_features(train_species_file, test_species_file):
    """
    Determines the common microbial feature columns between the training and validation species CSV files.
    Assumes that the species CSV files have a 'specimen' column and the rest of the columns are microbial features.
    
    Returns:
      A sorted list of common feature column names.
    """
    train_species = pd.read_csv(train_species_file)
    test_species = pd.read_csv(test_species_file)
    
    # Exclude the 'specimen' column
    train_feats = set(train_species.columns) - {'specimen'}
    test_feats = set(test_species.columns) - {'specimen'}
    common = sorted(list(train_feats.intersection(test_feats)))
    
    return common

def run_analysis(collect_wk_threshold, delivery_wk_threshold, outcome_name, species_train, species_test, meta_train, meta_test):
    """
    Runs the full analysis:
      - Loads and processes training and test data.
      - Restricts to common microbial features.
      - Fits a logistic regression model.
      - Evaluates the model on the test set by printing AUC and plotting the ROC curve.
    """
    # Process training and validation datasets
    train_df = load_and_process(species_train, meta_train, collect_wk_threshold, delivery_wk_threshold, outcome_name)
    test_df  = load_and_process(species_test,  meta_test,  collect_wk_threshold, delivery_wk_threshold, outcome_name)
    
    # Get common microbial feature names (from the original species abundance CSV files)
    common_features = get_common_features(species_train, species_test)
    print(f"Number of common microbial features: {len(common_features)}")
    
    # Prepare feature matrices and outcome vectors
    X_train = train_df[common_features]
    y_train = train_df[outcome_name]
    X_test  = test_df[common_features]
    y_test  = test_df[outcome_name]
    
    # Fit a logistic regression model (using regularization, increase max_iter if needed)
    clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict probabilities for the test set
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Calculate AUC ROC
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC ROC for {outcome_name} prediction: {auc:.3f}")
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {outcome_name} Prediction')
    plt.legend(loc="lower right")
    plt.savefig("./out2/o3-high-2.py_"+outcome_name+".pdf", format="pdf")
    df = pd.DataFrame({'y_test': y_test, 'y_prob': y_prob})
    df.to_csv("./out2/o3-high-2.py_"+outcome_name+".csv", index=False)


# =============================================================================
# Main analysis
# =============================================================================

# File paths
species_train = 'training_species_abundance.csv'
species_test  = 'validation_species_abundance.csv'
meta_train    = 'training_metadata.csv'
meta_test     = 'validation_metadata.csv'

# -------------------------------
# 1. Analysis for PTB (delivery_wk < 37)
#    Use the last specimen with collect_wk < 32 per participant.
# -------------------------------
print("Running analysis for PTB (delivery_wk < 37, specimens with collect_wk < 32):")
run_analysis(
    collect_wk_threshold=32,
    delivery_wk_threshold=37,
    outcome_name='PTB',
    species_train=species_train,
    species_test=species_test,
    meta_train=meta_train,
    meta_test=meta_test
)

# -------------------------------
# 2. Analysis for EarlyPTB (delivery_wk < 32)
#    Use the last specimen with collect_wk < 28 per participant.
# -------------------------------
print("\nRunning analysis for EarlyPTB (delivery_wk < 32, specimens with collect_wk < 28):")
run_analysis(
    collect_wk_threshold=28,
    delivery_wk_threshold=32,
    outcome_name='EarlyPTB',
    species_train=species_train,
    species_test=species_test,
    meta_train=meta_train,
    meta_test=meta_test
)
