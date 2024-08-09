import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Generate theoretical graphs for each control mode
n_samples = 100
x = np.linspace(0, 10, n_samples)

# Theoretical graphs
y_no_pv = -x  # Straight line negative slope
y_pv_volt_var = x + 10  # Straight line positive slope
y_pv_volt_watt = (x - 5) ** 2  # Quadratic minimum
y_pv_pf = -(x - 5) ** 2 + 25  # Quadratic maximum

# Generate new data to classify
new_data_x = np.linspace(0, 10, n_samples)
new_data_y = (new_data_x - 5) ** 2 + np.random.normal(0, 5, n_samples)  # Example new data

# Prepare training data
X_train = np.concatenate([y_no_pv, y_pv_volt_var, y_pv_volt_watt, y_pv_pf]).reshape(-1, 1)
y_train = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples + [3]*n_samples)

# Prepare new data for prediction
X_new = new_data_y.reshape(-1, 1)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict control mode for new data
predictions = rf_clf.predict(X_new)

# Determine the most frequent prediction as the final prediction
final_prediction = np.argmax(np.bincount(predictions))

# Print the final prediction
control_modes = ['No PV', 'PV+Volt-VAR', 'PV+Volt-Watt', 'PV+PF']
print(f"Final Predicted Control Mode: {control_modes[final_prediction]}")

# Plot the theoretical graphs and new data
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y_no_pv, label='No PV (Straight line negative)', color='blue')
plt.scatter(new_data_x, new_data_y, label='New Data', color='cyan', marker='x')
plt.title('No PV')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, y_pv_volt_var, label='PV+Volt-VAR (Straight line positive)', color='green')
plt.scatter(new_data_x, new_data_y, label='New Data', color='lightgreen', marker='x')
plt.title('PV+Volt-VAR')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, y_pv_volt_watt, label='PV+Volt-Watt (Quadratic minimum)', color='red')
plt.scatter(new_data_x, new_data_y, label='New Data', color='salmon', marker='x')
plt.title('PV+Volt-Watt')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, y_pv_pf, label='PV+PF (Quadratic maximum)', color='purple')
plt.scatter(new_data_x, new_data_y, label='New Data', color='violet', marker='x')
plt.title('PV+PF')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
print(classification_report(y_train, rf_clf.predict(X_train)))
