import numpy as np
import matplotlib.pyplot as plt

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

# Calculate distances
def calculate_total_distance(y_theoretical, y_new):
    return np.sum((y_theoretical - y_new) ** 2)

distances = {
    'No PV': calculate_total_distance(y_no_pv, new_data_y),
    'PV+Volt-VAR': calculate_total_distance(y_pv_volt_var, new_data_y),
    'PV+Volt-Watt': calculate_total_distance(y_pv_volt_watt, new_data_y),
    'PV+PF': calculate_total_distance(y_pv_pf, new_data_y)
}

# Select the mode with the smallest distance
final_prediction = min(distances, key=distances.get)

# Print the final prediction
print(f"Final Predicted Control Mode: {final_prediction}")

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
