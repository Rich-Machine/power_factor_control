using MAT
using Plots
using JSON

# Specify the path to your .mat file
file_path = "/Users/rasiamah3/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Research/Just Code/JuliaDevelopment/smart_meter_data/voltage_control_data.mat"

# Load the data from the .mat file
data = matread(file_path)

# # Access the active and reactive power in the loaded data
p = data["pvRealPower"]
q = data["pvReactivePower"]
PV_busNames = data["PV"]["busName"]
load_buses = data["Loads"]["busName"]
nodal_voltages = data["netloadV"]
nodal_p = data["netloadP"]
nodal_q = data["netloadQ"]

## Preprocessing to take out the PV inputs during the night when both P and Q are zero
## Assuming the sun rises and sets at 6am and 8pm respectively
global time_intervals = []
for i in 0 : 364
    global time_intervals = vcat(time_intervals, (i*96 + 25):(i*96 + 80))
end

global preprocessed_p = []
global preprocessed_q = []
global preprocessed_nodal_voltages = []
global preprocessed_nodal_p = []
global preprocessed_nodal_q = []

for i in time_intervals
    global preprocessed_p = vcat(preprocessed_p, p[i, :])
    global preprocessed_q = vcat(preprocessed_q, q[i, :])
    global preprocessed_nodal_voltages = vcat(preprocessed_nodal_voltages, nodal_voltages[i, :])
    global preprocessed_nodal_p = vcat(preprocessed_nodal_p, nodal_p[i, :])
    global preprocessed_nodal_q = vcat(preprocessed_nodal_q, nodal_q[i, :])
end

p = reshape(preprocessed_p, length(time_intervals), 701)
q = reshape(preprocessed_q, length(time_intervals), 701)
nodal_voltages = reshape(preprocessed_nodal_voltages, length(time_intervals), 1379)
nodal_p = reshape(preprocessed_nodal_p, length(time_intervals), 1379)
nodal_q = reshape(preprocessed_nodal_q, length(time_intervals), 1379)

preprocessed_data = Dict("p" => p, "q" => q, "nodal_voltages" => nodal_voltages, "nodal_p" => nodal_p, "nodal_q" => nodal_q)
## Save the preprocessed data to a JSON file\
open("preprocessed_data.json", "w") do f
    JSON.print(f, preprocessed_data)
end

# Of the 701 PV panels, which ones do you want to pots?
first_PV = 1
last_pv = 10

## This plots the active and reactive power of the PV panels.
for load in first_PV:last_pv
    real_power = []
    reactive_power = []
    voltage = []

    for i in 1:96*30 ## I choose one day, the first day of the year but can be changed to any day
        push!(real_power, p[i,load] * -1)
        push!(reactive_power, q[i, load])

        this_bus = PV_busNames[load]
        index = 1
        for busess in load_buses
            if busess == this_bus
                
               push!(voltage, nodal_voltages[i, index])
            end
            index = index + 1
        end
    end

    # Create a new plot for each PV panel
    plt = scatter(real_power, reactive_power, xlabel="Active Power", ylabel="Reactive Power", label=false, title="PV Panel $load")
    
    # Display the plot
    display(plt)

    plt = scatter(voltage, reactive_power, xlabel="Voltage", ylabel="Reactive Power", label=false, title="PV Panel $load")
    
    # Display the plot
    display(plt)

    plt = scatter(voltage, real_power, xlabel="Voltage", ylabel="Active Power", label=false, title="PV Panel $load")
    
    # Display the plot
    display(plt)
end

## This plots the bus voltage, active and reactive power injections.
for i in 1:10
    real_power = []
    reactive_power = []
    voltage = []

    push!(real_power, nodal_p[i, :])
    push!(reactive_power, nodal_q[i, :])
    push!(voltage, nodal_voltages[i, :])

    # Create a new plot for each PV panel
    plt = scatter(real_power, reactive_power, xlabel="Active Power", ylabel="Reactive Power", label=false, title="Bus $i")

    # Display the plot
    display(plt)

    plt = scatter(voltage, reactive_power, xlabel="Voltage", ylabel="Reactive Power", label=false, title="Bus $i")

    # Display the plot
    display(plt)

    plt = scatter(voltage, real_power, xlabel="Voltage", ylabel="Active Power", label=false, title="Bus $i")

    # Display the plot
    display(plt)
end