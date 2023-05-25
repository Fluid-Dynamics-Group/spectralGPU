module Integrate

using ..markers: AbstractParallel, AbstractState, AbstractWavenumbers, AbstractConfig, AbstractForcing, AbstractIoExport
using ..fft: ifftn_mpi!, fftn_mpi!
using ..mesh: Wavenumbers, WavenumbersGPU
using ..state: State, StateGPU
using ..config: Config, calculate_dt
using ..solver: compute_rhs! 
import ..Io

using CUDA

function integrate(
    parallel::P, 
    K::Wavenumbers, 
    config::Config{CFG}, 
    state::State,
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
    t = 0
    tstep = 0

    dt = calculate_dt(U, config)

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

        for exporter in io_exports
            stepper = Io.get_stepper(exporter)

            # if this exporter is intended on 
            if Io.should_write(stepper, t)
                Io.export_data(exporter, t)
                Io.increase_step!(stepper)
            end
        end
    end
end


#
end
