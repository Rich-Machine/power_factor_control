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

three_cases = [1, 3, 4,6,7,8,9]
for i in three_cases
    real_power = []
    reactive_power = []
    voltage = []

    for j in 1:96
        push!(real_power, nodal_p[j, i])
        push!(reactive_power, nodal_q[j, i])
        push!(voltage, nodal_voltages[j, i])
    end
    sum

    pf = real_power ./ sqrt.(real_power .^ 2 + reactive_power .^ 2)
    # display(maximum(pf))
    # display(minimum(pf))
    # display(pf)

    # Create a new plot for each PV panel
    plt = plot((1:96), real_power, xlabel="Time", ylabel="Active Power", label=false, title="Time variation of Active power at bus $i")

    # Display the plot
    display(plt)

    # Create a new plot for each PV panel
    plt = plot((1:96), reactive_power, xlabel="Time", ylabel="Reactive Power", label=false, title="Time variation of Reactive power at bus $i")

    # Display the plot
    display(plt)

    # # Create a new plot for each PV panel
    # plt = plot((1:96), voltage, xlabel="Time", ylabel="Voltage Magnitude", label=false, title="Time variation of voltage at bus $i")

    # # Display the plot
    # display(plt)
end

for i in 1:10
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
    # display(plt)

    # plt = plot((1:96), reactive_power, xlabel="Time", ylabel="Reactive Power", label=false, title="Time variation of Reactive power at Panel $i")
    
    # # Display the plot
    # display(plt)
end

# one_hot = Dict()
# one_hot_vector = zeros(1379)
# index_bus_with_pv = []
# for j in 1:1379
#     index = 1
#     for i in keys(PV_busNames)
#         if PV_busNames[i] == data["Loads"]["busName"][j]
#             push!(index_bus_with_pv, data["Loads"]["Idx"][j])
#             if data["PV"]["controlMode"][index] == "VOLTVAR"
#                 one_hot_vector[j] = 2  
#             else
#                 one_hot_vector[j] = 3
#             end
#         end
#         index= index + 1
#     end
# end
# for i in 1:1379
#     if !(i in index_bus_with_pv)
#         one_hot_vector[i] = 1
#     end
# end

# json_data = JSON.json(one_hot_vector)

# # Specify the path to save the JSON file
# json_file_path = "one_hot_vector.json"

# # Write the JSON data to the file
# open(json_file_path, "w") do file
#     write(file, json_data)
# end
