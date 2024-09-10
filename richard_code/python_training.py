##Installing packages in VS Code
# python3 -m venv env
# source env/bin/activate
import tensorflow as tf
import numpy as np
import mat73
import json
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

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

nodal_voltages = nodal_voltages.reshape(48320160,)
nodal_p = nodal_p.reshape(48320160,)
nodal_q = nodal_q.reshape(48320160,)

x = np.concatenate((nodal_p, nodal_q, nodal_voltages), axis=0)
print(x.shape)
x = x.reshape(48320160, 3)
print(x.shape)

y = to_categorical(one_hot_vector)
y = np.repeat(y, 35040, axis=0)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Definition
# Get parameters from logged hyperparameters
model = Sequential([
  Flatten(input_shape=(3, )),
  Dense(128, activation='relu'),
  Dense(64, activation='relu'),
  Dropout(0.25),
  Dense(4, activation='softmax')
  ])

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adamax', 
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train,
    batch_size=50,
    epochs=50,
    validation_data=(x_test, y_test),)

score = model.evaluate(x_test, y_test)
predicts = model.predict(x_test)
print(predicts)
print(y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

