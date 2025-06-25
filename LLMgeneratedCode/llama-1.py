import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Download GSE149440 dataset from GEO
url = 'https://www.ncbi.nlm.nih.gov/geo/query.cgi?acc=GSE149440&submit=Submit+query'
response = requests.get(url)

# Parse the XML response and extract the expression data
data = pd.read_csv('GSE149440-GEODATA.txt', sep='\t')

# Map gene symbols to their IDs
gene_map = {row['symbol']: row['accession'] for index, row in data.iterrows()}
gene_ids = [gene_map[genename] for genenames in data['feature'].values]

# Get the metadata and create a DataFrame with relevant information
metadata_url = 'https://www.ncbi.nlm.nih.gov/geo/query.cgi?acc=GSE149440&submit=Submit+query'
response = requests.get(metadata_url)
metadata = pd.read_csv('metadata.txt', sep='\t')

# Extract the gestational age data and filter out samples not assigned to training set
gestational_age = metadata['gestation']
train_samples = metadata[metadata['train:ch1'] == '1']['accession'].values

# Split data into features (gene expression) and target variable (gestational age)
X = pd.DataFrame(data[['feature', gene_ids]].T, columns=data['feature']).set_index('accession')
y = gestational_age.loc[train_samples].reset_index(drop=True)

# Fit a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X.loc[X.index.isin(train_samples)], 
                                                    y.loc[y.index.isin(train_samples)], 
                                                    test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate root mean squared error (RMSE) of predictions
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
print('Root Mean Squared Error:', rmse)

# Generate scatter plot for prediction vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Gestational Age')
plt.ylabel('Predicted Gestational Age')
plt.title('Gestational Age Prediction Scatter Plot')
plt.savefig("./out2/llama-1.py.pdf", format="pdf")

