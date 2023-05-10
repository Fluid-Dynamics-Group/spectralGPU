module config

using ..markers: AbstractConfig

export Config, create_config, taylor_green_validation, calculate_dt

struct Config <: AbstractConfig
    ν::Float64
    re::Float64
    N::Int
    time::Float64
end

struct ConfigValidation <: AbstractConfig
    ν::Float64
    re::Float64
    N::Int
    time::Float64
end

function create_config(N::Int, re::Float64, time::Float64)::Config
    ν = 1 / re

    CFL_NUM = 0.75

    return Config(ν, re, N, time)
end

function taylor_green_validation()::ConfigValidation
    N = 64
    re = 1600
    ν = 1 / re

    t = 0.1

    return ConfigValidation(ν, re, N, t)
end

function calculate_dt(velocity::XARRAY, cfg::Config)::Float64 where XARRAY <: AbstractArray{Float64, 4}
    max_velo = maximum(velocity)
    CFL = 0.75

    return CFL / (cfg.N * max_velo)
end

function calculate_dt(velocity::XARRAY, cfg::ConfigValidation)::Float64 where XARRAY <: AbstractArray{Float64, 4}
    return 0.01
end

#
end
