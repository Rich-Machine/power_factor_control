% Load the data file
load('Ckt5_Case2VV_NetPQV_AllCust_20210330T185329.mat');

% Extract relevant data from the loaded variables
V = netloadV;  % Voltage
Q = netloadQ;  % Reactive Power
P = netloadP;  % Real Power

% Time vector (assuming 15-minute intervals)
time = (0:15:(size(V, 1) - 1) * 15) / 60 / 24; % Time in days

% Plot the time series data for Real Power, Reactive Power, and Voltage
figure;

% Real Power (P)
subplot(3, 1, 1);
plot(time, P, 'r', 'LineWidth', 1);
xlabel('Time (days)');
ylabel('Real Power (P)');
title('Real Power (P) over Time');

% Reactive Power (Q)
subplot(3, 1, 2);
plot(time, Q, 'b', 'LineWidth', 1);
xlabel('Time (days)');
ylabel('Reactive Power (Q)');
title('Reactive Power (Q) over Time');

% Voltage (V)
subplot(3, 1, 3);
plot(time, V, 'g', 'LineWidth', 1);
xlabel('Time (days)');
ylabel('Voltage (V)');
title('Voltage (V) over Time');

% Adjust layout
sgtitle('Time Series Data for Real Power, Reactive Power, and Voltage');
