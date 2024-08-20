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

three_cases = [1,2,8]
for i in three_cases
    real_power = []
    reactive_power = []
    voltage = []

    for j in 1:96
        push!(real_power, nodal_p[j, i])
        push!(reactive_power, nodal_q[j, i])
        push!(voltage, nodal_voltages[j, i])
    end

    pf = real_power ./ sqrt.(real_power .^ 2 + reactive_power .^ 2)
    display(maximum(pf))
    display(minimum(pf))
    # display(pf)

    # Create a new plot for each PV panel
    plt = plot((1:96), real_power, xlabel="Time", ylabel="Active Power", label=false, title="Time variation of Active power at bus $i")

    # Display the plot
    display(plt)

    # Create a new plot for each PV panel
    plt = plot((1:96), reactive_power, xlabel="Time", ylabel="Reactive Power", label=false, title="Time variation of Reactive power at bus $i")

    # Display the plot
    display(plt)

    # Create a new plot for each PV panel
    plt = plot((1:96), voltage, xlabel="Time", ylabel="Voltage Magnitude", label=false, title="Time variation of voltage at bus $i")

    # Display the plot
    display(plt)
end

for i in 1:2
    real_power = []
    reactive_power = []
    voltage = []

    for j in 1:96 ## I choose one day, the first day of the year but can be changed to any day
        push!(real_power, p[j, i] * -1)
        push!(reactive_power, q[j, i])
    end

    # Create a new plot for each PV panel
    plt = plot((1:96), real_power, xlabel="Time", ylabel="Active Power", label=false, title="Time variation of Active power at Panel $i")

    # Display the plot
    display(plt)

    plt = plot((1:96), reactive_power, xlabel="Time", ylabel="Reactive Power", label=false, title="Time variation of Reactive power at Panel $i")
    
    # Display the plot
    display(plt)
end

index_bus_with_pv = []
for j in 1:1379
    for i in keys(PV_busNames)
        if PV_busNames[i] == data["Loads"]["busName"][j]
            push!(index_bus_with_pv, data["Loads"]["Idx"][j])
        end
    end
end