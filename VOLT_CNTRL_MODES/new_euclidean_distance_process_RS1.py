import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Function to scale, normalize, and shift the graphs
def preprocess_graph(graph, common_baseline=0, scale_size=1):
    scaler = MinMaxScaler()
    graph_scaled = scaler.fit_transform(graph.reshape(-1, 1)).flatten() * scale_size
    graph_shifted = graph_scaled - np.mean(graph_scaled) + common_baseline
    return graph_shifted

# Generate theoretical graphs for each control mode
n_samples = 100
x = np.linspace(0, 10, n_samples)

y_no_pv = -x  # Straight line negative slope
y_pv_volt_var = x + 10  # Straight line positive slope
y_pv_volt_watt = (x - 5) ** 2  # Quadratic minimum
y_pv_pf = -(x - 5) ** 2 + 25  # Quadratic maximum

# Preprocess the theoretical graphs
scale_size = 10  # Define a common scale size
y_no_pv_preprocessed = preprocess_graph(y_no_pv, scale_size=scale_size)
y_pv_volt_var_preprocessed = preprocess_graph(y_pv_volt_var, scale_size=scale_size)
y_pv_volt_watt_preprocessed = preprocess_graph(y_pv_volt_watt, scale_size=scale_size)
y_pv_pf_preprocessed = preprocess_graph(y_pv_pf, scale_size=scale_size)

# Generate new data to classify
new_data_x = np.linspace(0, 10, n_samples)
new_data_y = (new_data_x - 5) ** 2 + np.random.normal(0, 5, n_samples)  # Example new data

# Preprocess the new data
new_data_y_preprocessed = preprocess_graph(new_data_y, scale_size=scale_size)

# Function to calculate Euclidean distance
def calculate_euclidean_distance(data, reference):
    return np.linalg.norm(data - reference)

# Calculate distances
distances = {
    'No PV': calculate_euclidean_distance(new_data_y_preprocessed, y_no_pv_preprocessed),
    'PV+Volt-VAR': calculate_euclidean_distance(new_data_y_preprocessed, y_pv_volt_var_preprocessed),
    'PV+Volt-Watt': calculate_euclidean_distance(new_data_y_preprocessed, y_pv_volt_watt_preprocessed),
    'PV+PF': calculate_euclidean_distance(new_data_y_preprocessed, y_pv_pf_preprocessed)
}

# Find the mode with the minimum distance
predicted_mode = min(distances, key=distances.get)
print(f"Predicted Control Mode: {predicted_mode}")

# Plot the preprocessed graphs and new data
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y_no_pv_preprocessed, label='No PV (Preprocessed)', color='blue')
plt.scatter(new_data_x, new_data_y_preprocessed, label='New Data (Preprocessed)', color='cyan', marker='x')
plt.title('No PV')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, y_pv_volt_var_preprocessed, label='PV+Volt-VAR (Preprocessed)', color='green')
plt.scatter(new_data_x, new_data_y_preprocessed, label='New Data (Preprocessed)', color='lightgreen', marker='x')
plt.title('PV+Volt-VAR')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, y_pv_volt_watt_preprocessed, label='PV+Volt-Watt (Preprocessed)', color='red')
plt.scatter(new_data_x, new_data_y_preprocessed, label='New Data (Preprocessed)', color='salmon', marker='x')
plt.title('PV+Volt-Watt')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, y_pv_pf_preprocessed, label='PV+PF (Preprocessed)', color='purple')
plt.scatter(new_data_x, new_data_y_preprocessed, label='New Data (Preprocessed)', color='violet', marker='x')
plt.title('PV+PF')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()
