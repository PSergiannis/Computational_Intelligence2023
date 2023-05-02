import pandas as pd

# Read in the data
df = pd.read_csv('preprocessed_dataset.csv')

# Extract the last column as the target variable
y = df.iloc[:, -1]

# Center all columns except the last column
X_centered = df.iloc[:, :-1] - df.iloc[:, :-1].mean()

# Concatenate the centered feature matrix and target variable
df_centered = pd.concat([X_centered, y], axis=1)

# Save the centered dataset to a new CSV file
df_centered.to_csv('centered_dataset.csv', index=False)


### Code to validate that the centering has been executed as expected
# # Calculate the sum of each column
# column_sums = df_centered.sum().round(decimals=1)
# # Print the column sums
# print(column_sums)