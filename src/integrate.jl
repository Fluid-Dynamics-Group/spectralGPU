module Integrate

using ..markers: AbstractParallel
using ..fft: ifftn_mpi!, fftn_mpi!
using ..mesh: Wavenumbers
using ..state: State
using ..config: Config
using ..solver: compute_rhs! 

function integrate(
    parallel::P, 
    K::Wavenumbers, 
    config::Config, 
    state::State,
    U::Array{Float64, 4}, 
    U_hat::Array{ComplexF64, 4}
) where P<: AbstractParallel
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
