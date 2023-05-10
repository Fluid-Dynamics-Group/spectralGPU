module config

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

function create_config(N::Int, re::Float64, time::Float64)::Config{ProductionConfig}
    ν = 1 / re
    return Config(ν, re, N, time, ProductionConfig())
end

function taylor_green_validation()::Config{ValidationConfig}
    N = 64
    re = 1600.
    ν = 1 / re

    t = 0.1

    return Config(ν, re, N, t, ValidationConfig())
end

cfl_calc(max_velo::Float64, N::Int) = 0.75 / (N * max_velo)

function calculate_dt(velocity::XARRAY, cfg::Config{ProductionConfig})::Float64 where XARRAY <: AbstractArray{Float64, 4}
    max_velo = maximum(velocity)
    return cfl_calc(max_velo, cfg.N)
end

#
# Validation dt cases
#
function calculate_dt(velocity::XARRAY, cfg::Config{ValidationConfig})::Float64 where XARRAY <: AbstractArray{Float64, 4}
    return 0.01
end

#
end
