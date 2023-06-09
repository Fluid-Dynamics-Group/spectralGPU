module Configuration

using ..markers: AbstractConfig, ProductionConfig, ValidationConfig

export Config, create_config, taylor_green_validation, calculate_dt

using CUDA

struct Config{C <: AbstractConfig}
    ν::Float64
    re::Float64
    N::Int
    time::Float64
    mode::C
end

function create_config(N::Int, re::Float64, time::Float64, U::ARR)::Config{ProductionConfig} where ARR <: AbstractArray{Float64, 4}
    ν = 1 / re
    max_velocity = maximum(abs.(U))
    return Config(ν, re, N, time, ProductionConfig(max_velocity))
end

# initialize a config with a hard-set maximum velocity
function create_config(N::Int, re::Float64, time::Float64, max_velocity::Float64)::Config{ProductionConfig}
    ν = 1 / re
    return Config(ν, re, N, time, ProductionConfig(max_velocity))
end

function taylor_green_validation()::Config{ValidationConfig}
    N = 64
    re = 1600.
    ν = 1 / re

    t = 0.1

    return Config(ν, re, N, t, ValidationConfig(0.01))
end

cfl_calc(max_velo::Float64, N::Int) = 0.75 / (N * max_velo)

function calculate_dt(cfg::Config{ProductionConfig})::Float64
    max_velo = cfg.mode.max_velocity
    return cfl_calc(max_velo, cfg.N)
end

#
# Validation dt cases
#
function calculate_dt(cfg::Config{ValidationConfig})::Float64
    return cfg.mode.fixed_dt
end

#
end
