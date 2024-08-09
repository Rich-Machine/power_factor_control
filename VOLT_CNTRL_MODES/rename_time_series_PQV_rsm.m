function plotTimeSeriesData(period)
    % Load the data file
    load('Ckt5_Case2VV_NetPQV_AllCust_20210330T185329.mat');

    % Extract relevant data from the loaded variables
    V = netloadV;  % Voltage
    Q = netloadQ;  % Reactive Power
    P = netloadP;  % Real Power

    % Time vector (assuming 15-minute intervals)
    time = (0:15:(size(V, 1) - 1) * 15) / 60 / 24; % Time in days

    % Define the sample index based on the chosen period
    switch period
        case '1 day'
            sample_index = 1:(24 * 4);
        case '3 days'
            sample_index = 1:(24 * 4 * 3);
        case '1 week'
            sample_index = 1:(24 * 4 * 7);
        case '2 weeks'
            sample_index = 1:(24 * 4 * 14);
        case '1 month'
            sample_index = 1:(24 * 4 * 30);
        case '2 months'
            sample_index = 1:(24 * 4 * 60);
        otherwise
            error('Invalid period specified. Choose from: 1 day, 3 days, 1 week, 2 weeks, 1 month, 2 months.');
    end

    % Ensure the sample index does not exceed the length of the data
    sample_index = sample_index(sample_index <= length(time));

    % Plot the time series data for Real Power, Reactive Power, and Voltage
    figure;

    % Real Power (P)
    subplot(3, 1, 1);
    plot(time(sample_index), P(sample_index), 'r', 'LineWidth', 1);
    xlabel('Time (days)');
    ylabel('Real Power (P)');
    title('Real Power (P) over Time');

    % Reactive Power (Q)
    subplot(3, 1, 2);
    plot(time(sample_index), Q(sample_index), 'b', 'LineWidth', 1);
    xlabel('Time (days)');
    ylabel('Reactive Power (Q)');
    title('Reactive Power (Q) over Time');

    % Voltage (V)
    subplot(3, 1, 3);
    plot(time(sample_index), V(sample_index), 'g', 'LineWidth', 1);
    xlabel('Time (days)');
    ylabel('Voltage (V)');
    title('Voltage (V) over Time');

    % Adjust layout
    sgtitle(['Time Series Data for Real Power, Reactive Power, and Voltage (', period, ')']);
end


plotTimeSeriesData('1 day');
plotTimeSeriesData('1 week');
plotTimeSeriesData('1 month');
