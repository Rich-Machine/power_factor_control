import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Generate synthetic data for each control mode
x = np.linspace(0, 10, 400)

# Volt-VAR control
V_L, V_1, V_2, V_ref, V_3, V_4, V_H = 2, 4, 5, 6, 7, 8, 10
Q_1, Q_2, Q_3, Q_4 = 1, 0.5, 0, -0.5
volt_var_y = np.piecewise(x, [x <= V_L, (x > V_L) & (x <= V_1), (x > V_1) & (x <= V_2), (x > V_2) & (x <= V_ref), (x > V_ref) & (x <= V_3), (x > V_3) & (x <= V_4), x > V_4],
                          [Q_1, lambda x: Q_1 - (Q_1 - Q_2) / (V_1 - V_L) * (x - V_L), Q_2, lambda x: Q_2 - (Q_2 - Q_3) / (V_ref - V_2) * (x - V_2), Q_3, lambda x: Q_3 - (Q_3 - Q_4) / (V_4 - V_3) * (x - V_3), Q_4])

# Volt-Watt control
P_1, P_2, P_3 = 1, 0.5, 0.1
volt_watt_y = np.piecewise(x, [x <= 3, (x > 3) & (x <= 6), (x > 6) & (x <= 9), x > 9],
                           [P_1, lambda x: P_1 - (P_1 - P_2) / (6 - 3) * (x - 3), lambda x: P_2 - (P_2 - P_3) / (9 - 6) * (x - 6), P_3])

# Power Factor control
pf = 0.9
power_factor_y = np.tan(np.arccos(pf)) * x

# No PV control
no_pv_y = np.random.random(400)

# Generate new data resembling one of the modes with noise
noise_level = 0.05
new_data_y = volt_watt_y + noise_level * np.random.randn(400)

# Combine all control modes into one dataset
X = np.vstack((volt_var_y, volt_watt_y, power_factor_y, no_pv_y)).T
y = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100)

# Split into training and testing sets
X_train, X_test = X[:300], X[300:]
y_train, y_test = y[:300], y[300:]

# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Prepare new data for prediction
new_data_X = np.vstack((new_data_y, new_data_y, new_data_y, new_data_y)).T

# Predict on the new data
new_data_pred = rf_clf.predict(new_data_X)

# Print the final prediction for the new data
print("Final Predicted Control Mode for New Data:", new_data_pred[0])

# Evaluate the model on the test set
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, volt_var_y, label='Volt-VAR (Synthetic)', color='blue')
plt.scatter(x, new_data_y, label='New Data', color='red')
plt.title('Volt-VAR')
plt.xlabel('X')
plt.ylabel('Q')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, volt_watt_y, label='Volt-Watt (Synthetic)', color='black')
plt.scatter(x, new_data_y, label='New Data', color='red')
plt.title('Volt-Watt')
plt.xlabel('X')
plt.ylabel('P')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, power_factor_y, label='Power Factor (Synthetic)', color='green')
plt.scatter(x, new_data_y, label='New Data', color='red')
plt.title('Power Factor')
plt.xlabel('X')
plt.ylabel('Q')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(x, no_pv_y, label='No PV (Synthetic)', color='blue')
plt.scatter(x, new_data_y, label='New Data', color='red')
plt.title('No PV')
plt.xlabel('X')
plt.ylabel('Q')
plt.legend()

plt.tight_layout()
plt.show()
