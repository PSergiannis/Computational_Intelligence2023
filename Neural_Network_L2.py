import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from keras.regularizers import l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the dataset
filename = 'standardized_dataset.csv'
data = pd.read_csv(filename)
print(len(data))

# Separate input features and target variable
X = data.drop('class', axis=1)
y = data['class']

# Set up 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True)

# Initialize results
results = {'accuracy': [], 'mse': [], 'ce': []}
train_history = {'loss': [], 'accuracy': []}

# Create neural network model
model = Sequential()
model.add(Dense(11, input_dim=17, activation='relu', kernel_regularizer=l2(0.9)))
model.add(Dense(5, activation='softmax', kernel_regularizer=l2(0.9)))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(learning_rate=0.001, momentum=0.2), metrics=['accuracy'])

X_train = X.iloc[[i for i in range(0, int(len(X) * 0.8))]]
y_train = y.iloc[[i for i in range(0, int(len(X) * 0.8))]]

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32)

# Save training history
train_history['loss'].extend(history.history['loss'])
train_history['accuracy'].extend(history.history['accuracy'])

# Loop through the folds
for train_index, test_index in tqdm(cv.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    y_pred_proba = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    ce = log_loss(y_test, y_pred_proba)

    results['accuracy'].append(accuracy)
    results['mse'].append(mse)
    results['ce'].append(ce)

# Calculate average results
avg_accuracy = np.mean(results['accuracy'])
avg_mse = np.mean(results['mse'])
avg_ce = np.mean(results['ce'])

print(f"Average accuracy: {avg_accuracy}")
print(f"Average MSE: {avg_mse}")
print(f"Average CE: {avg_ce}")

# Plot training loss and accuracy
plt.figure()
plt.plot(train_history['loss'], label='Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(train_history['accuracy'], label='Accuracy')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
