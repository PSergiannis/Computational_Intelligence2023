import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the CSV file
df = pd.read_csv('dataset-HAR-PUC-Rio.csv', sep=';')



# Replace the values in the gender column
df['gender'] = df['gender'].map({'Man': 1, 'Woman': 0})

# Map the values in the class column to integers
df['class'] = df['class'].map({'sittingdown': 0, 'standingup': 1, 'standing': 2, 'walking': 3, 'sitting': 4})

# Convert 'how_tall_in_meters' and 'body_mass_index' columns to float
df['how_tall_in_meters'] = df['how_tall_in_meters'].str.replace(',', '.').astype(float)
df['body_mass_index'] = df['body_mass_index'].str.replace(',', '.').astype(float)

# Drop the user column
df.drop('user', axis=1, inplace=True)

# # Filter the data to remove any record that has a non-integer z4 value
# df = df[df['z4'].apply(lambda x: isinstance(x, int))]

# Convert 'z4' column to int
df['z4'] = pd.to_numeric(df['z4'], errors='coerce').astype('Int64')
df.dropna(inplace=True)

# Save the modified dataset to a new CSV file
df.to_csv('preprocessed_dataset.csv', index=False)

print(df.dtypes)
print(len(df))

# # Center each column at 0
# df_centered = df - df.mean()

# # Save the modified dataset to a new CSV file
# df_centered.to_csv('modified_dataset_centered.csv', index=False)




# # Split the data into the class column and the rest of the columns
# X = df.iloc[:, :-1]  # all columns except the last one
# y = df.iloc[:, -1]   # last column only

# # Center the data
# centered = X - X.mean()

# # # Normalize the data
# # normalized = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=df.columns)

# # # Standardize the data
# # standardized = pd.DataFrame(StandardScaler().fit_transform(X), columns=df.columns)

# # Concatenate the data with the class column
# centered_with_class = pd.concat([centered, y], axis=1)
# # normalized_with_class = pd.concat([normalized, y], axis=1)
# # standardized_with_class = pd.concat([standardized, y], axis=1)

# # Save the data to CSV files
# centered_with_class.to_csv('centered_with_class.csv', index=False)
# # normalized_with_class.to_csv('normalized_with_class.csv', index=False)
# # standardized_with_class.to_csv('standardized_with_class.csv', index=False)

# # Save the data to CSV files
# centered_with_class.to_csv('centered_with_class.csv', index=False)
# # normalized_with_class.to_csv('normalized_with_class.csv', index=False)
# # standardized_with_class.to_csv('standardized_with_class.csv', index=False)




