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
last_pv = 100

for load in first_PV:last_pv
    real_power = []
    reactive_power = []
    voltage = []

    for i in 1:96 ## I choose one day, the first day of the year but can be changed to any day
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

    apparent_power = .√(real_power.^2 + reactive_power.^2)
    power_factor = real_power ./ apparent_power
    for  i in eachindex(power_factor)
        if isnan(power_factor[i])
            power_factor[i] = 1
        end
    end
    q_over_v = reactive_power ./  voltage
    p_over_v = real_power ./  voltage
    tolerance = 0.05

    if count(x -> all(abs(x - y) <= tolerance for y in power_factor), power_factor) >= 0.99 * length(power_factor)
        println("Load $load is Fixed Power Factor controlled.")
    end
    if count(x -> x == 0 || all(abs(x - y) <= tolerance for y in q_over_v), q_over_v) >= 0.99 * length(q_over_v)
        println("Load $load is Volt-VAR controlled.")
    end
    if count(x -> x == 0 || all(abs(x - y) <= tolerance for y in p_over_v), p_over_v) >= 0.95 * length(p_over_v)
        println("Load $load is Volt-Watt controlled.")
    end
end

