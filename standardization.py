from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the CSV file
df = pd.read_csv('preprocessed_dataset.csv')

# Drop the last column ('class')
X = df.drop('class', axis=1)

# Scale the data
scaler = StandardScaler()
X_centered = scaler.fit_transform(X)

# Convert back to a DataFrame
df_centered = pd.DataFrame(X_centered, columns=X.columns)

# Add the 'class' column back to the DataFrame
df_centered['class'] = df['class']

# Save the centered dataset to a new CSV file
df_centered.to_csv('standardized_dataset.csv', index=False)
