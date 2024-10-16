import tensorflow as tf
import numpy as np
import mat73
import json
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)
params = {
    'dropout': 0.25,
    'batch-size': 128,
    'epochs': 50,
    'layer-1-size': 128,
    'layer-2-size': 128,
    'initial-lr': 0.01,
    'decay-steps': 2000,
    'decay-rate': 0.9,
    'optimizer': 'adamax'
}

# Load the json file
with open('one_hot_vector.json') as f:
    one_hot_vector = json.load(f)

# Specify the path to your .mat file
file_path = "/Users/rasiamah3/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Research/Just Code/JuliaDevelopment/smart_meter_data/voltage_control_data.mat"

# Load the data from the .mat file
data = mat73.loadmat(file_path)

# Access the active and reactive power in the loaded data
nodal_voltages = data["netloadV"]
nodal_p = data["netloadP"]
nodal_q = data["netloadQ"]

nodal_p = np.transpose(nodal_p)
nodal_q = np.transpose(nodal_q)
nodal_voltages = np.transpose(nodal_voltages)

nodal_voltages = nodal_voltages.reshape(48320160,1)
nodal_p = nodal_p.reshape(48320160,1)
nodal_q = nodal_q.reshape(48320160,1)

x = np.concatenate((nodal_p, nodal_q, nodal_voltages), axis=1)
# x= np.transpose(x)
print(x.shape)
# x = x.reshape(503335, 288)
# print(x.shape)

# Assuming each sample should be 96 timesteps long
timesteps = 96
num_samples = x.shape[0] // timesteps  # Number of samples with 96 timesteps each

x = x[:num_samples * timesteps]  # Trim the data to fit exactly into (num_samples, 96, 3)
x = x.reshape(num_samples, timesteps, 3)
print("New input shape:", x.shape)

# One-hot encoding the target variable
y = to_categorical(one_hot_vector)
y = np.repeat(y, 365, axis=0)
y = y.reshape(num_samples, 1, 4)
print("New target shape:", y.shape)
# y = np.repeat(y, 365, axis=0)
# # y = np.repeat(y, 35040, axis=0)
# print(y.shape)

# Splitting the reshaped data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Remove the extra dimension in the target labels
y_train = y_train.reshape(-1, 4)
y_test = y_test.reshape(-1, 4)

# Model Definition
# Get parameters from logged hyperparameters
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(96, 3)))  # Input: (96, 3)
model.add(Dropout(0.25))
model.add(Flatten())  # Flatten the output before the final Dense layer
model.add(Dense(4, activation='softmax')) 

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model.compile(optimizer='adamax', 
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=50,
          epochs=50,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping])

score = model.evaluate(x_test, y_test)
predicts = model.predict(x_test)

rounded_predicts = np.round(predicts)
accuracy = np.mean(rounded_predicts == y_test)
error = np.mean(rounded_predicts != y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))


