module solver

export curl!, cross!, compute_rhs!

using ..markers: AbstractParallel, AbstractState, AbstractWavenumbers, AbstractConfig
using ..fft: ifftn_mpi!, fftn_mpi!, Plan
using ..mesh: Wavenumbers, WavenumbersGPU
using ..state: State, StateGPU
using ..config: Config

using CUDA

# calculate the curl of a fourier space array `input` and store the result in the x-space array `out`
@views function __curl!(
    parallel::P, 
    K::WAVE, 
    plan::Plan,
    input::FARRAY
    ;
    out::XARRAY,
    tmp::FARRAY3
) where P <: AbstractParallel where FARRAY <: AbstractArray{ComplexF64, 4} where XARRAY <: AbstractArray{Float64, 4} where WAVE <: AbstractWavenumbers where FARRAY3 <: AbstractArray{ComplexF64, 3}
    j = complex(0, 1)

    tmp[:, :, :] .= K[1].*input[:, :, :, 2] 
    tmp[:, :, :] .-= K[2].*input[:, :, :, 1]
    tmp .*= j
    ifftn_mpi!(parallel, K, plan, tmp, out[:, :, :, 3])

    tmp[:, :, :] .= K[3].*input[:, :, :, 1] 
    tmp[:, :, :] .-=  K[1].*input[:, :, :, 3]
    tmp[:, :, :] .*= j

    ifftn_mpi!(parallel, K, plan, tmp, out[:, :, :, 2])

    tmp[:, :, :] .= K[2].*input[:, :, :, 3] 
    tmp[:, :, :] .-= K[3].*input[:, :, :, 2]
    tmp[:, :, :] .*= j
    ifftn_mpi!(parallel, K, plan, tmp, out[:, :, :, 1])

    nothing
end

function curl!(
    parallel::P, 
    K::Wavenumbers, 
    plan::Plan,
    input::Array{ComplexF64, 4}
    ;
    out::Array{Float64, 4},
    tmp::Array{ComplexF64, 3}
) where P <: AbstractParallel
    __curl!(parallel, K, plan, input; out = out, tmp=tmp);
end

function curl!(
    parallel::P, 
    K::WavenumbersGPU,
    plan::Plan,
    input::CuArray{ComplexF64, 4}
    ;
    out::CuArray{Float64, 4},
    tmp::CuArray{ComplexF64, 3}
) where P <: AbstractParallel
    __curl!(parallel, K, plan, input; out = out, tmp = tmp);
end


# take the cross product of X-Space a,b vectors 
# stores the result in (fourier space) `out` and then return the
@views function __cross!(
    parallel::P, 
    plan::Plan,
    a::XARRAY,
    b::XARRAY
    ;
    out::FARRAY,
    tmp::XARRAY3
) where P <: AbstractParallel where FARRAY <: AbstractArray{ComplexF64, 4} where XARRAY <: AbstractArray{Float64, 4} where XARRAY3 <: AbstractArray{Float64, 3}
    tmp[:, :, :] .= a[:, :, :, 2].*b[:, :, :, 3]
    tmp[:, :, :] .-= a[:, :, :, 3].*b[:, :, :, 2]
    fftn_mpi!(parallel, plan, tmp, out[:, :, :, 1])

    tmp[:, :, :] .= a[:, :, :, 3].*b[:, :, :, 1]
    tmp[:, :, :] .-= a[:, :, :, 1].*b[:, :, :, 3]
    fftn_mpi!(parallel, plan, tmp, out[:, :, :, 2])

    tmp[:, :, :] .= a[:, :, :, 1].*b[:, :, :, 2]
    tmp[:, :, :] .-= a[:, :, :, 2].*b[:, :, :, 1]
    fftn_mpi!(parallel, plan, tmp, out[:, :, :, 3])

    nothing
end

function cross!(
    parallel::P, 
    plan::Plan,
    a::Array{Float64, 4},
    b::Array{Float64, 4},
    ;
    out::Array{ComplexF64, 4},
    tmp::Array{Float64, 3},
) where P <: AbstractParallel
    __cross!(parallel, plan, a, b, out = out, tmp=tmp);

    nothing
end

function cross!(
    parallel::P, 
    plan::Plan,
    a::CuArray{Float64, 4},
    b::CuArray{Float64, 4},
    ;
    out::CuArray{ComplexF64, 4},
    tmp::CuArray{Float64, 3},
) where P <: AbstractParallel
    __cross!(parallel, plan, a, b, out = out, tmp=tmp);
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
    U::CuArray{Float64, 4},
    U_hat::CuArray{ComplexF64, 4},
    state::StateGPU
) where P <: AbstractParallel
    __compute_rhs!(rk_step, parallel, K, U, U_hat, state)
end

function compute_rhs!(
    rk_step::Int,
    parallel::P,
    K::Wavenumbers,
    U::Array{Float64, 4},
    U_hat::Array{ComplexF64, 4},
    state::State
) where P <: AbstractParallel
    __compute_rhs!(rk_step, parallel, K, U, U_hat, state)
end

function __compute_rhs!(
    rk_step::Int,
    parallel::P,
    K::WAVE,
    U::ARRAY,
    U_hat::FARRAY,
    state::STATE
) where P <: AbstractParallel where FARRAY <: AbstractArray{ComplexF64, 4} where ARRAY <: AbstractArray{Float64, 4} where STATE <: AbstractState where WAVE <: AbstractWavenumbers

    if rk_step != 1
        @views for i in 1:3
            ifftn_mpi!(parallel, K, state.fft_plan, U_hat[:, :, :, i], U[:, :, :, i])
        end
    end

    curl!(parallel, K, state.fft_plan, U_hat; out = state.curl, tmp=state.curl_tmp)
    cross!(parallel, state.fft_plan, U, state.curl; out = state.dU, tmp = state.cross_tmp)
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

    state.dU .-= state.ν .* state.K² .* U_hat
end

#
end
