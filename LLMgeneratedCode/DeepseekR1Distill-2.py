import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# Read training data
train_species = pd.read_csv('training_species_abundance.csv')
val_species = pd.read_csv('validation_species_abundance.csv')

# Read metadata
train_meta = pd.read_csv('training_metadata.csv')
val_meta = pd.read_csv('validation_metadata.csv')

def process_data(merged_df, collect_wk_threshold):
    return merged_df.groupby('participant_id').apply(
        lambda x: x[x['collect_wk'] <= collect_wk_threshold].sort_values('collect_wk', ascending=False).iloc[0]
    )

# Process PTB (delivery_wk <37)
def process_ptb_data():
    # Train set processing
    train_ptb = pd.merge(train_species, train_meta, on='participant_id', how='left')
    train_ptb = process_data(train_ptb, 32)
    train_ptb['PTB'] = (train_ptb['delivery_wk'] < 37).astype(int)
    
    # Validation set processing
    val_ptb = pd.merge(val_species, val_meta, on='participant_id', how='left')
    val_ptb = process_data(val_ptb, 32)
    val_ptb['PTB'] = (val_ptb['delivery_wk'] < 37).astype(int)
    
    common_features_ptb = set(train_species.columns) & set(val_species.columns) - {'participant_id'}
    train_ptb = train_ptb[common_features_ptb]
    val_ptb = val_ptb[common_features_ptb]
    
    # Split participants
    participants = list(train_ptb.index.get_level_values('participant_id'))
    np.random.shuffle(participants)
    split_idx = int(0.8 * len(participants))
    train_participants = participants[:split_idx]
    test_participants = participants[split_idx:]
    
    X_train_ptb = train_ptb.loc[train_participants, common_features_ptb].values
    y_train_ptb = train_ptb.loc[train_participants, 'PTB'].values
    X_test_ptb = val_ptb.loc[test_participants, common_features_ptb].values
    y_test_ptb = val_ptb.loc[test_participants, 'PTB'].values
    
    # Fit model
    model_ptb = LogisticRegression()
    model_ptb.fit(X_train_ptb, y_train_ptb)
    
    # Predict and evaluate
    y_pred_ptb = model_ptb.predict_proba(X_test_ptb)[:, 1]
    roc_auc = roc_auc_score(y_test_ptb, y_pred_ptb)
    print(f"AUC-ROC for PTB: {roc_auc}")
    plt.figure()
    RocCurveDisplay.from_predictions(y_test_ptb, y_pred_ptb)
    plt.savefig("./out2/DeepseekDistill-2.py_A.pdf", format="pdf")
	
    
process_ptb_data()

def process_earlyptb_data():
    # Train set processing
    train_earlyptb = pd.merge(train_species, train_meta, on='participant_id', how='left')
    train_earlyptb = process_data(train_earlyptb, 28)
    train_earlyptb['EarlyPTB'] = (train_earlyptb['delivery_wk'] < 32).astype(int)
    
    # Validation set processing
    val_earlyptb = pd.merge(val_species, val_meta, on='participant_id', how='left')
    val_earlyptb = process_data(val_earlyptb, 28)
    val_earlyptb['EarlyPTB'] = (val_earlyptb['delivery_wk'] < 32).astype(int)
    
    common_features_earlyptb = set(train_species.columns) & set(val_species.columns) - {'participant_id'}
    train_earlyptb = train_earlyptb[common_features_earlyptb]
    val_earlyptb = val_earlyptb[common_features_earlyptb]
    
    participants = list(train_earlyptb.index.get_level_values('participant_id'))
    np.random.shuffle(participants)
    split_idx = int(0.8 * len(participants))
    train_participants = participants[:split_idx]
    test_participants = participants[split_idx:]
    
    X_train_earlyptb = train_earlyptb.loc[train_participants, common_features_earlyptb].values
    y_train_earlyptb = train_earlyptb.loc[train_participants, 'EarlyPTB'].values
    X_test_earlyptb = val_earlyptb.loc[test_participants, common_features_earlyptb].values
    y_test_earlyptb = val_earlyptb.loc[test_participants, 'EarlyPTB'].values
    
    # Fit model
    model_earlyptb = LogisticRegression()
    model_earlyptb.fit(X_train_earlyptb, y_train_earlyptb)
    
    # Predict and evaluate
    y_pred_earlyptb = model_earlyptb.predict_proba(X_test_earlyptb)[:, 1]
    roc_auc = roc_auc_score(y_test_earlyptb, y_pred_earlyptb)
    print(f"AUC-ROC for EarlyPTB: {roc_auc}")
    plt.figure()
    RocCurveDisplay.from_predictions(y_test_earlyptb, y_pred_earlyptb)
    plt.savefig("./out2/DeepseekDistill-2.py_B.pdf", format="pdf")
process_earlyptb_data()
