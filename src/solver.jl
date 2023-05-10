module solver

export curl!, cross!, compute_rhs!

using ..markers: AbstractParallel, AbstractState, AbstractWavenumbers
using ..fft: ifftn_mpi!, fftn_mpi!
using ..mesh: Wavenumbers, WavenumbersGPU
using ..state: State, StateGPU
using ..config: Config

using CUDA

# calculate the curl of a fourier space array `input` and store the result in the x-space array `out`
@views function curl!(
    parallel::P, 
    K::Wavenumbers, 
    input::Array{ComplexF64, 4}
    ;
    out::Array{Float64, 4}
) where P <: AbstractParallel
    j = complex(0, 1)

    ifftn_mpi!(parallel, K, j*(K[1].*input[:, :, :, 2] .- K[2].*input[:, :, :, 1]), out[:, :, :, 3])
    ifftn_mpi!(parallel, K, j*(K[3].*input[:, :, :, 1] .- K[1].*input[:, :, :, 3]), out[:, :, :, 2])
    ifftn_mpi!(parallel, K, j*(K[2].*input[:, :, :, 3] .- K[3].*input[:, :, :, 2]), out[:, :, :, 1])

    nothing
end

@views function curl!(
    parallel::P, 
    K::WavenumbersGPU,
    input::CuArray{ComplexF64, 4}
    ;
    out::CuArray{Float64, 4}
) where P <: AbstractParallel
    j = complex(0, 1)

    @CUDA.sync ifftn_mpi!(parallel, K,j * (CuArray(K[1]).*input[:, :, :, 2] .- CuArray(K[2]).*input[:, :, :, 1]), out[:, :, :, 3])
    @CUDA.sync ifftn_mpi!(parallel, K, j*(K[3].*input[:, :, :, 1] .- K[1].*input[:, :, :, 3]), out[:, :, :, 2])
    @CUDA.sync ifftn_mpi!(parallel, K, j*(K[2].*input[:, :, :, 3] .- K[3].*input[:, :, :, 2]), out[:, :, :, 1])

    nothing
end


# take the cross product of X-Space a,b vectors 
# stores the result in (fourier space) `out` and then return the
@views function cross!(
    parallel::P, 
    a::Array{Float64, 4},
    b::Array{Float64, 4},
    ;
    out::Array{ComplexF64, 4}
) where P <: AbstractParallel
    fftn_mpi!(parallel, a[:, :, :, 2].*b[:, :, :, 3] .- a[:, :, :, 3].*b[:, :, :, 2], out[:, :, :, 1])
    fftn_mpi!(parallel, a[:, :, :, 3].*b[:, :, :, 1] .- a[:, :, :, 1].*b[:, :, :, 3], out[:, :, :, 2])
    fftn_mpi!(parallel, a[:, :, :, 1].*b[:, :, :, 2] .- a[:, :, :, 2].*b[:, :, :, 1], out[:, :, :, 3])

    nothing
end

@views function cross!(
    parallel::P, 
    a::CuArray{Float64, 4},
    b::CuArray{Float64, 4},
    ;
    out::CuArray{ComplexF64, 4}
) where P <: AbstractParallel
    @CUDA.sync fftn_mpi!(parallel, a[:, :, :, 2].*b[:, :, :, 3] .- a[:, :, :, 3].*b[:, :, :, 2], out[:, :, :, 1])
    @CUDA.sync fftn_mpi!(parallel, a[:, :, :, 3].*b[:, :, :, 1] .- a[:, :, :, 1].*b[:, :, :, 3], out[:, :, :, 2])
    @CUDA.sync fftn_mpi!(parallel, a[:, :, :, 1].*b[:, :, :, 2] .- a[:, :, :, 2].*b[:, :, :, 1], out[:, :, :, 3])

    nothing
end

function wavenumber_product!(arr::Array{ComplexF64, 3}, K::Wavenumbers; out::Array{ComplexF64, 4})
    out[:, :, :, 1] .= arr .* K.kx
    out[:, :, :, 2] .= arr .* K.ky
    out[:, :, :, 3] .= arr .* K.kz
end

function wavenumber_product!(arr::CuArray{ComplexF64, 3}, K::WavenumbersGPU; out::CuArray{ComplexF64, 4})
    out[:, :, :, 1] .= arr .* K.kx
    out[:, :, :, 2] .= arr .* K.ky
    out[:, :, :, 3] .= arr .* K.kz
end

function compute_rhs!(
    rk_step::Int,
    parallel::P,
    K::WavenumbersGPU,
    config::Config,
    U::CuArray{Float64, 4},
    U_hat::CuArray{ComplexF64, 4},
    state::StateGPU
) where P <: AbstractParallel
    __compute_rhs!(rk_step, parallel, K, config, U, U_hat, state)
end

function compute_rhs!(
    rk_step::Int,
    parallel::P,
    K::Wavenumbers,
    config::Config,
    U::Array{Float64, 4},
    U_hat::Array{ComplexF64, 4},
    state::State
) where P <: AbstractParallel
    __compute_rhs!(rk_step, parallel, K, config, U, U_hat, state)
end

function __compute_rhs!(
    rk_step::Int,
    parallel::P,
    K::WAVE,
    config::Config,
    U::ARRAY,
    U_hat::FARRAY,
    state::STATE
) where P <: AbstractParallel where FARRAY <: AbstractArray{ComplexF64, 4} where ARRAY <: AbstractArray{Float64, 4} where STATE <: AbstractState where WAVE <: AbstractWavenumbers

    if rk_step != 1
        @views for i in 1:3
            ifftn_mpi!(parallel, K, U_hat[:, :, :, i], U[:, :, :, i])
        end
    end

    curl!(parallel, K, U_hat; out = state.curl)
    cross!(parallel, U, state.curl; out = state.dU)
    state.dU .*= state.dealias

    state.P_hat[:, :, :] .= complex(0., 0);
    # compute P_hat = sum(dU * K_over_K²):
    sum!(state.P_hat, state.dU .* state.K_over_K²)

    # TODO: make any adjustments to the forcing vector
    
    # add Pressure term to dU/dt
    wavenumber_product!(state.P_hat, K; out = state.wavenumber_product_tmp)
    state.dU .-= state.wavenumber_product_tmp

    # dU is now the Eulerian term / Nonlinear term in the NSE
    # now calculate the diffusion term

    state.dU .-= config.ν .* state.K² .* U_hat
end

#
end
