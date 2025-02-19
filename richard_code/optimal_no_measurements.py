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
    # Subtract 1 from each element in one_hot_vector
    one_hot_vector = [x - 1 for x in one_hot_vector]

# Specify the path to your .mat file
file_path = "/Users/rasiamah3/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Research/just_code/JuliaDevelopment/smart_meter_data/voltage_control_data.mat"

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
f1_score_values = []
accuracy_values = []

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
    model.add(Dense(2 * timesteps, activation='relu', input_shape=(timesteps, 3)))
    model.add(Dense(timesteps, activation='relu'))
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

    # Calculate recall and precision
    print(classification_report(y_test, y_pred))
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('F1 Score: {:.2f}'.format(f1_score))

    f1_score_values.append(f1_score)
    accuracy_values.append(accuracy)

# Save the vectors f1 and accuracy
np.save('f1_score_values.npy', f1_score_values)
np.save('accuracy_values.npy', accuracy_values)

# Plotting recall vs timesteps
plt.figure(figsize=(8, 6))
plt.plot(all_timesteps, f1_score_values, marker='o')
plt.xlabel('Timesteps')
plt.ylabel('F1 Score')
plt.title('F1 score VS Number of Timesteps')
plt.savefig('F1_score_vs_timesteps.png')
plt.show()

# Plotting accuracy vs timesteps
plt.figure(figsize=(8, 6))
plt.plot(all_timesteps, accuracy_values, marker='*', color='red')
plt.xlabel('Timesteps')
plt.ylabel('Accuracy')
plt.title('Accuracy VS Number of Timesteps')
plt.grid(True)
plt.savefig('accuracy_vs_timesteps.png')
plt.show()



        
# # Generate predictions on the test set
# y_pred_cm = np.argmax(y_pred, axis=1)
# y_test_cm = np.argmax(y_test, axis=1)
# # Compute the confusion matrix
# cm = confusion_matrix(y_test_cm, y_pred_cm)
# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# # Compute the false positive rate, true positive rate, and thresholds for the roc curve
# fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
# roc_auc = auc(fpr, tpr)
# # Plot the roc curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc='lower right')
# plt.show()
