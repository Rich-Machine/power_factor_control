load = 11
# for load in first_PV:last_pv
    real_power = []
    reactive_power = []

    for i in 1:96 ## I choose one day, the first day of the year but can be changed to any day
        push!(real_power, p[i,load] * -1)
        push!(reactive_power, q[i, load])
    end
    plot(real_power, reactive_power)