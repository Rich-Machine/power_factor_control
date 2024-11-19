import tensorflow as tf
import numpy as np
import mat73
import json
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import roc_curve, auc
import h5py
import matplotlib.pyplot as plt

# Set random seed
tf.random.set_seed(43)
np.random.seed(43)
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
    # Subtract 1 from each element in one_hot_vector
    one_hot_vector = [x - 1 for x in one_hot_vector]

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

# Assuming each sample should be 96 timesteps long
all_timesteps = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96]

# Plotting recall and accuracy vs timesteps
accuracy_values = []
random_accuracy_values = []

for timesteps in all_timesteps:
    print(f"Training model with {timesteps} timesteps")
    x = np.concatenate((nodal_p, nodal_q, nodal_voltages), axis=1)
    print(x.shape)
    num_samples = x.shape[0] // timesteps  # Number of samples with 96 timesteps each

    x = x[:num_samples * timesteps]  # Trim the data to fit exactly into (num_samples, 96, 3)
    x = x.reshape(num_samples, timesteps, 3)
    print("New input shape:", x.shape)

    # One-hot encoding the target variable
    y = to_categorical(one_hot_vector)
    y = np.repeat(y, 35040/timesteps, axis=0)
    print("New target shape:", y.shape)

    # Splitting the reshaped data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Model Definition
    model = Sequential()
    model.add(Dense(timesteps, activation='relu', input_shape=(timesteps, 3)))
    # model.add(Dense(timesteps, activation='relu'))
    model.add(Flatten())  # Flatten the output before the final Dense layer
    model.add(Dense(3, activation='softmax')) 

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model.compile(optimizer='adamax', 
                loss=loss_fn,
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=25,
            epochs = 100,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping])

    # Generate predictions on the test set
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)
    accuracy = np.mean(y_pred == y_test)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    accuracy_values.append(accuracy)

    # Shuffle the rows in x_test and y_test in the same order
    random_indices = np.random.permutation(len(x_test[1,:,1]))
    x_test = x_test[:, random_indices, :]
    # y_test = y_test[random_indices]

    # Generate predictions on the test set
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)
    random_accuracy = np.mean(y_pred == y_test)
    print('Random Accuracy: {:.2f}%'.format(random_accuracy * 100))
    random_accuracy_values.append(random_accuracy)

# Plotting accuracy vs timesteps
plt.figure(figsize=(8, 6))
plt.plot(all_timesteps, accuracy_values, marker='o')
plt.plot(all_timesteps, random_accuracy_values, marker='*')
plt.legend(['Consecutive', 'Random'])
plt.xlabel('Timesteps')
plt.ylabel('Accuracy')
plt.title('Accuracy of Random VS Consecutive sampling')
plt.grid(True)  # Add grid lines to the plot
plt.savefig('random_vs_consecutive.png')
plt.show()
