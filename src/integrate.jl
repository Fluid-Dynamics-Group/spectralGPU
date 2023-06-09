module Integrate

using ..Markers: AbstractParallel, AbstractState, AbstractWavenumbers, AbstractConfig, AbstractForcing, AbstractIoExport
using ..Fft: ifftn_mpi!, fftn_mpi!
using ..Mesh: Wavenumbers, WavenumbersGPU
using ..State: StateCPU, StateGPU
using ..Configuration: Config, calculate_dt
using ..Solver: compute_rhs!, curl!
import ..Io

using CUDA

function integrate(
    parallel::P, 
    K::Wavenumbers, 
    config::Config{CFG},
    state::StateCPU,
    U::Array{Float64, 4}, 
    U_hat::Array{ComplexF64, 4},
    forcing::FORCING,
    io_exports::Vector{AbstractIoExport},
) where P<: AbstractParallel where CFG  <: AbstractConfig where FORCING <: AbstractForcing
    __integrate(parallel, K, config, state, U, U_hat, forcing, io_exports)
end

function integrate(
    parallel::P, 
    K::WavenumbersGPU,
    config::Config{CFG},
    state::StateGPU,
    U::CuArray{Float64, 4}, 
    U_hat::CuArray{ComplexF64, 4},
    forcing::FORCING,
    io_exports::Vector{AbstractIoExport},
) where P<: AbstractParallel where CFG  <: AbstractConfig where FORCING <: AbstractForcing
    __integrate(parallel, K, config, state, U, U_hat, forcing, io_exports)
end

function __integrate(
    parallel::P,
    K::WAVE,
    config::Config{CFG},
    state::STATE,
    U::XARRAY,
    U_hat::FARRAY,
    forcing::FORCING,
    io_exports::Vector{AbstractIoExport},
) where P<: AbstractParallel where FARRAY <: AbstractArray{ComplexF64, 4} where XARRAY <: AbstractArray{Float64, 4} where STATE <: AbstractState where WAVE <: AbstractWavenumbers where CFG  <: AbstractConfig where FORCING <: AbstractForcing
    t::Float64 = 0.
    tstep::Int = 0

    dt = calculate_dt(config)

    # initialize the vorticity before we do any writes
    curl!(parallel, K, state.fft_plan, U_hat; out = state.curl)

    Io.write_io(t, io_exports);

    while t < config.time - 1e-8
        t += dt
        tstep += 1

        state.U_hat₁[:, :, :, :] .= U_hat[:, :, :, :]
        state.U_hat₀[:, :, :, :] .= U_hat[:, :, :, :]

        for rk_step in 1:4
            compute_rhs!(rk_step, parallel, K, U, U_hat, state, forcing)

            #println("du abs change: ", sum(abs.(state.dU)))

            if rk_step < 4
                U_hat[:, :, :, :] .= state.U_hat₀ .+ state.b[rk_step] * dt * state.dU
            end

            state.U_hat₁ .+= state.a[rk_step] * dt .* state.dU
        end

        U_hat[:, :, :, :] .= state.U_hat₁

        # update U from the rk time integrated U_hat
        @views for i in 1:3
            ifftn_mpi!(parallel, K, state.fft_plan, U_hat[:, :, :, i], U[:, :, :, i])
        end

        Io.write_io(t, io_exports);
    end
end

#
end
