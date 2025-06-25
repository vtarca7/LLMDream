import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import requests
import zipfile
import io

# Step 1: Download the dataset from GEO
def download_gse_data(gse_id):
    base_url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[3:]}/'
    gsm_list_url = f'{base_url}GSM{"/".join([f"{int(i):05d}" for i in gse_id.split("_")[1:]]).replace("0", "")}.txt.gz'
    
    # Download metadata
    meta_response = requests.get(f'{base_url}{gse_id}.soft.gz')
    with open(f'{gse_id}.soft', 'wb') as f:
        f.write(meta_response.content)
    
    # Extract GSM IDs from the soft file
    gsm_ids = []
    with gzip.open(f'{gse_id}.soft', 'rt') as f:
        for line in f:
            if line.startswith('!Series_matrix_table_begin'):
                break
        else:
            raise ValueError("Could not find Series matrix table begin")
        
        for line in f:
            if line.startswith('ID_REF'):
                cols = line.strip().split('\t')
                continue
            elif cols and len(line.split('\t')) > 0:
                gsm_ids.append(line.strip().split('\t')[0])
    
    # Download expression data for each GSM ID
    for gsm_id in gsm_ids:
        expr_url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[3:]}/{gsm_id[:5]}/GSM{gsm_id}.soft.gz'
        response = requests.get(expr_url)
        with open(f'{gsm_id}.soft', 'wb') as f:
            f.write(response.content)

def load_data(gse_id):
    # Load metadata
    meta_df = pd.read_table(f'{gse_id}.soft', comment='!', header=0, sep='\t')
    
    # Extract the necessary columns from metadata
    meta_df = meta_df[['ID_REF', 'characteristics_ch1.1']]
    meta_df['ID_REF'] = meta_df['ID_REF'].apply(lambda x: int(x.split('.')[2]))
    meta_df.columns = ['GSM_ID', 'gestational_age']
    
    # Extract train and gestational age information
    train_info = pd.DataFrame(meta_df['characteristics_ch1.1'].str.extract(r'train:ch1=([\d]+);.+?(\d+) weeks'))
    train_info.columns = ['train', 'gestational_age']
    train_info['GSM_ID'] = meta_df['GSM_ID']
    
    # Convert gestational age to float
    train_info['gestational_age'] = train_info['gestational_age'].astype(float)
    train_info['train'] = train_info['train'].astype(int)
    
    return train_info

def load_expression_data(gse_id):
    expr_dfs = []
    for file in os.listdir('.'):
        if file.endswith('soft') and 'GSM' in file:
            gsm_id = int(file.split('.')[0].split('_')[1])
            expr_df = pd.read_table(file, comment='!', header=2)
            expr_df['GSM_ID'] = gsm_id
            expr_dfs.append(expr_df)
    
    # Combine all expression dataframes
    full_expr_df = pd.concat(expr_dfs)
    
    return full_expr_df

def main():
    gse_id = 'GSE149440'
    download_gse_data(gse_id)

    train_info = load_data(gse_id)
    expr_data = load_expression_data(gse_id)

    # Merge metadata with expression data
    merged_data = expr_data.merge(train_info, on='GSM_ID')
    
    # Split into training and testing sets
    train_set = merged_data[merged_data['train'] == 1]
    test_set = merged_data[merged_data['train'] == 0]

    X_train = train_set.drop(columns=['GSM_ID', 'gestational_age', 'train'])
    y_train = train_set['gestational_age']
    
    X_test = test_set.drop(columns=['GSM_ID', 'gestational_age', 'train'])
    y_test = test_set['gestational_age']
    
    # Fit a regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    predictions = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f'Root Mean Squared Error (RMSE) on the Test Set: {rmse:.2f}')
    
    # Generate scatter plot of predicted vs actual gestational ages
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Gestational Age')
    plt.ylabel('Predicted Gestational Age')
    plt.title('Predicted vs Actual Gestational Ages')
    plt.savefig("./out2/phi4-1.py.pdf", format="pdf")

if __name__ == '__main__':
    main()
