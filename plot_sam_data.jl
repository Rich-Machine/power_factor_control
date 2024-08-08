using MAT
using Plots

# Specify the path to your .mat file
file_path = "voltage_control_data.mat"

# Load the data from the .mat file
data = matread(file_path)

# # Access the active and reactive power in the loaded data
p = data["pvRealPower"]
q = data["pvReactivePower"]
PV_busNames = data["PV"]["busName"]
load_buses = data["Loads"]["busName"]
nodal_voltages = data["netloadV"]

# Of the 701 PV panels, which ones do you want to pots?
first_PV = 1
last_pv = 10

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