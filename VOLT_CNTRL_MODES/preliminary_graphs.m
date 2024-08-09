% Load the data file
load('Ckt5_Case2VV_NetPQV_AllCust_20210330T185329.mat');

% Extract relevant data from the loaded variables
V = netloadV;  % Voltage
Q = netloadQ;  % Reactive Power
P = netloadP;  % Real Power

% Sample index for demonstration (you can adjust this as needed)
sample_index = 1:100;  % Adjust the range as needed

% Volt-VAR Control
figure;
subplot(3, 1, 1);
plot(V(sample_index), Q(sample_index), 'b', 'LineWidth', 1.5);
xlabel('Voltage (V)');
ylabel('Reactive Power (Q)');
title('Volt-VAR Control');

% Volt-Watt Control
subplot(3, 1, 2);
plot(V(sample_index), P(sample_index), 'r', 'LineWidth', 1.5);
xlabel('Voltage (V)');
ylabel('Real Power (P)');
title('Volt-Watt Control');

% Power Factor Control
pf = 0.9;  % Assuming a power factor of 0.9 for illustration
subplot(3, 1, 3);
plot(P(sample_index), Q(sample_index), 'g', 'LineWidth', 1.5);
xlabel('Real Power (P)');
ylabel('Reactive Power (Q)');
title('Power Factor Control');

% Adjust layout
sgtitle('Control Modes');
