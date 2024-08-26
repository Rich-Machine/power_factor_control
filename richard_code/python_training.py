##Installing packages in VS Code
# python3 -m venv env
# source env/bin/activate

# first neural network with keras tutorial
from numpy import loadtxt
import scipy
import mat73
import json
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os



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
input_data = [nodal_p, nodal_q, nodal_voltages]
# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(35040*3,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
print(model.summary())
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(input_data, one_hot_vector, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

