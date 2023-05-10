module Integrate

using ..markers: AbstractParallel, AbstractState, AbstractWavenumbers
using ..fft: ifftn_mpi!, fftn_mpi!
using ..mesh: Wavenumbers, WavenumbersGPU
using ..state: State, StateGPU
using ..config: Config
using ..solver: compute_rhs! 

using CUDA

function integrate(
    parallel::P, 
    K::Wavenumbers, 
    config::Config, 
    state::State,
    U::Array{Float64, 4}, 
    U_hat::Array{ComplexF64, 4}
) where P<: AbstractParallel
    __integrate(parallel, K, config, state, U, U_hat)
end

function integrate(
    parallel::P, 
    K::WavenumbersGPU,
    config::Config,
    state::StateGPU,
    U::CuArray{Float64, 4}, 
    U_hat::CuArray{ComplexF64, 4}
) where P<: AbstractParallel
    __integrate(parallel, K, config, state, U, U_hat)
end

function __integrate(
    parallel::P, 
    K::WAVE, 
    config::Config, 
    state::STATE,
    U::XARRAY,
    U_hat::FARRAY,
) where P<: AbstractParallel where FARRAY <: AbstractArray{ComplexF64, 4} where XARRAY <: AbstractArray{Float64, 4} where STATE <: AbstractState where WAVE <: AbstractWavenumbers
    t = 0
    tstep = 0

    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples
    # a_j is equal to the canonical b_i
    # b_i is equal to the canonical a_j
    a = [1/6, 1/3, 1/3, 1/6]
    b = [0.5, 0.5, 1.]

    while t < config.time - 1e-8
        println("stepping");

        t += config.dt
        tstep += 1

        state.U_hat₁[:, :, :, :] .= U_hat[:, :, :, :]
        state.U_hat₀[:, :, :, :] .= U_hat[:, :, :, :]

        for rk_step in 1:4
            compute_rhs!(rk_step, parallel, K, config, U, U_hat, state)

            println("du abs change: ", sum(abs.(state.dU)))

            if rk_step < 4
                U_hat[:, :, :, :] .= state.U_hat₀ .+ b[rk_step] * config.dt * state.dU
            end

            state.U_hat₁ .+= a[rk_step] * config.dt .* state.dU
        end

        U_hat[:, :, :, :] .= state.U_hat₁

        # update U from the rk time integrated U_hat
        @views for i in 1:3
            ifftn_mpi!(parallel, K, U_hat[:, :, :, i], U[:, :, :, i])
        end
    end
end


#
end
