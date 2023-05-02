from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Read in the data
df = pd.read_csv('preprocessed_dataset.csv')

# Extract the last column as the target variable
y = df.iloc[:, -1]

# Normalize all columns except the last column
scaler = MinMaxScaler()
X = scaler.fit_transform(df.iloc[:, :-1])

# Concatenate the normalized feature matrix and target variable
df_normalized = pd.concat([pd.DataFrame(X), y], axis=1)

# Set column names of the normalized dataframe
df_normalized.columns = df.columns

# Save the centered dataset to a new CSV file
df_normalized.to_csv('normalized_dataset.csv', index=False)