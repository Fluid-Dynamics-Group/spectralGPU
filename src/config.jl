module config

export Config, create_config, taylor_green_validation

struct Config
    ν::Float64
    re::Float64
    N::Int
    time::Float64
    dt::Float64
end

function create_config(N::Int, re::Float64, time::Float64)::Config
    ν = 1 / re

    dt = 0.01

    return Config(ν, re, N, time, dt)
end

function taylor_green_validation()::Config
    N = 64
    re = 1600
    ν = 1 / re

    t = 0.1
    dt = 0.01

    return Config(ν, re, N, t, dt)
end

end
