using MAT
using Plots
using StatsBase
using Distances

# Specify the path to your .mat file
file_path = "/Users/rasiamah3/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Research/Just Code/JuliaDevelopment/smart_meter_data/voltage_control_data.mat"

# Load the data from the .mat file
data = matread(file_path)

# # Access the active and reactive power in the loaded data
p = data["netloadP"]
q = data["netloadQ"]
PV_busNames = data["PV"]["busName"]
load_buses = data["Loads"]["busName"]
nodal_voltages = data["netloadV"]

no_control = []
pf_control = []
vv_control = []
index_bus_with_pv = []

for j in 1:1100
    index = 1
    for i in keys(PV_busNames)
        if PV_busNames[i] == data["Loads"]["busName"][j]
            push!(index_bus_with_pv, data["Loads"]["Idx"][j])
            if data["PV"]["controlMode"][index] == "VOLTVAR"
                push!(vv_control, j) 
            else
                push!(pf_control, j)
            end
        end
        index= index + 1
    end
end
for i in 1:1100
    if !(i in index_bus_with_pv)
        push!(no_control, i)
    end
end

avg_p_no = []
avg_q_no = []
normalized_p = p[:, no_control] ./ maximum(p[:, no_control])
normalized_q = q[:, no_control] ./ maximum(p[:, no_control])
for i in 1:96
    sum_p = sum(sum(normalized_p[96 * (day - 1) + i, j] for day in 1:365) for j in size(no_control))
    sum_q = sum(sum(normalized_q[96 * (day - 1) + i, j] for day in 1:365) for j in size(no_control))
    push!(avg_p_no, sum_p/length(no_control))
    push!(avg_q_no, sum_q/length(no_control))
end
avg_p_no = convert(Vector{Float64}, avg_p_no)
avg_q_no = convert(Vector{Float64}, avg_q_no)
plt = plot(1:96, avg_p_no, label="Average Real Power", xlabel="Time (15-min intervals)", ylabel="Power (kW)", title="Waveform for No Control")
plt = plot!(1:96, avg_q_no, label="Average Reactive Power")
display(plt)

avg_p_pf = []
avg_q_pf = []
normalized_p = p[:, pf_control] ./ maximum(p[:, pf_control])
normalized_q = q[:, pf_control] ./ maximum(p[:, pf_control])
for i in 1:96
    sum_p = sum(sum(normalized_p[96 * (day - 1) + i, j] for day in 1:365) for j in size(pf_control))
    sum_q = sum(sum(normalized_q[96 * (day - 1) + i, j] for day in 1:365) for j in size(pf_control))
    push!(avg_p_pf, sum_p/length(pf_control))
    push!(avg_q_pf, sum_q/length(pf_control))
end
avg_p_pf = convert(Vector{Float64}, avg_p_pf)
avg_q_pf = convert(Vector{Float64}, avg_q_pf)
plt = plot(1:96, avg_p_pf, label="Average Real Power", xlabel="Time (15-min intervals)", ylabel="Power (kW)", title="Waveform for PF Control")
plt = plot!(1:96, avg_q_pf, label="Average Reactive Power")
display(plt)

avg_p_vv = []
avg_q_vv = []
normalized_p = p[:, vv_control] ./ maximum(p[:, vv_control])
normalized_q = q[:, vv_control] ./ maximum(p[:, vv_control])
for i in 1:96
    sum_p = sum(sum(normalized_p[96 * (day - 1) + i, j] for day in 1:365) for j in size(vv_control))
    sum_q = sum(sum(normalized_q[96 * (day - 1) + i, j] for day in 1:365) for j in size(vv_control))
    push!(avg_p_vv, sum_p/length(vv_control))
    push!(avg_q_vv, sum_q/length(vv_control))
end
avg_p_vv = convert(Vector{Float64}, avg_p_vv)
avg_q_vv = convert(Vector{Float64}, avg_q_vv)
plt = plot(1:96, avg_p_vv, label="Average Real Power", xlabel="Time (15-min intervals)", ylabel="Power (kW)", title="Waveform for Volt-Var Control")
plt = plot!(1:96, avg_q_vv, label="Average Reactive Power")
display(plt)

for j in 1:1379
    index = 1
    for i in keys(PV_busNames)
        if PV_busNames[i] == data["Loads"]["busName"][j]
            push!(index_bus_with_pv, data["Loads"]["Idx"][j])
            if data["PV"]["controlMode"][index] == "VOLTVAR"
                push!(vv_control, j) 
            else
                push!(pf_control, j)
            end
        end
        index= index + 1
    end
end
for i in 1:1379
    if !(i in index_bus_with_pv)
        push!(no_control, i)
    end
end

#Specify the day of the year you are looking at
global count = 0
global overall_predictions = 0
global undetermined = 0
global correct_predictions = 0
for load in 1101:1379

    no_control_number = 0
    pf_control_number = 0
    vv_control_number = 0
    for day in 1:365
        # ## Adding noise to the testing dataset.
        # for i in 1:96
        #     p[96 * (day - 1) + i, load] = p[96 * (day - 1) + i, load] .+ rand(-1:1) .* (0.05 * mean(p[96 * (day - 1) + i, load]))
        #     q[96 * (day - 1) + i, load] = q[96 * (day - 1) + i, load] .+ rand(-1:1) .* (0.05 * mean(q[96 * (day - 1) + i, load]))
        # end

        count = count + 1
        normalized_p_values = p[96 * (day - 1) + 1 : 96 * (day - 1) + 96, load] ./ maximum(p[96 * (day - 1) + 1 : 96 * (day - 1) + 96, load]) 
        normalized_q_values = q[96 * (day - 1) + 1 : 96 * (day - 1) + 96, load] ./ maximum(p[96 * (day - 1) + 1 : 96 * (day - 1) + 96, load]) 
        dist_p_no = euclidean(normalized_p_values, avg_p_no)
        dist_p_pf = euclidean(normalized_p_values, avg_p_pf)
        dist_p_vv = euclidean(normalized_p_values, avg_p_vv)
        dist_q_no = euclidean(normalized_q_values, avg_q_no)
        dist_q_pf = euclidean(normalized_q_values, avg_q_pf)
        dist_q_vv = euclidean(normalized_q_values, avg_q_vv)

        if dist_p_no < dist_p_pf && dist_p_no < dist_p_vv ##&& dist_q_vv > dist_q_no && dist_q_vv > dist_q_pf
            # println("The load at bus $load is best controlled with no control")
            no_control_number += 1
            if load in no_control
                correct_predictions += 1
            end
        elseif dist_q_vv < dist_q_no && dist_q_vv < dist_q_pf ##&& dist_p_no > dist_p_vv && dist_p_no > dist_p_pf
            # println("The load at bus $load is best controlled with Volt-Var control")
            vv_control_number += 1
            if load in vv_control
                correct_predictions += 1
            end
        elseif dist_p_no > dist_p_pf && dist_q_vv > dist_q_pf ##&& dist_p_no > dist_p_vv && dist_q_vv > dist_q_no && dist_q_vv > dist_q_pf
            # println("The load at bus $load is best controlled with PF control")
            pf_control_number += 1
            if load in pf_control
                correct_predictions += 1
            end
        # else
            undetermined += 1
        #     println("The control mode at bus $load cannot be determined")
        end
    end
    max_control = findmax([no_control_number, pf_control_number, vv_control_number])
    if max_control[2] == 1 && load in no_control
        overall_predictions =+ 1
    elseif max_control[2] == 2 && load in pf_control
        overall_predictions =+ 1
    elseif max_control[2] == 3 && load in vv_control
        overall_predictions =+ 1
    end
end

accuracy = round((correct_predictions / count) * 100, digits=2)
println("The accuracy of the model on an daily level is $accuracy%")

accuracy = round((undetermined/ count) * 100, digits=2)
println("The percentage of samples that are undetermined are $accuracy%")