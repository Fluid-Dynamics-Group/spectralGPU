module solver

export curl!, cross!, compute_rhs!

using ..markers: AbstractParallel, AbstractState, AbstractWavenumbers, AbstractConfig, AbstractForcing
using ..fft: ifftn_mpi!, fftn_mpi!, Plan
using ..mesh: Wavenumbers, WavenumbersGPU
using ..state: State, StateGPU
using ..config: Config
using ..Forcing: force_system!

using CUDA

# calculate the curl of a fourier space array `input` and store the result in the x-space array `out`
@views function __curl!(
    parallel::P, 
    K::WAVE, 
    plan::Plan,
    input::FARRAY
    ;
    out::XARRAY
) where P <: AbstractParallel where FARRAY <: AbstractArray{ComplexF64, 4} where XARRAY <: AbstractArray{Float64, 4} where WAVE <: AbstractWavenumbers
    j::ComplexF64 = complex(0., 1.)

    ifftn_mpi!(parallel, K, plan, j.*(K[2].*input[:, :, :, 3] .- K[3].*input[:, :, :, 2]), out[:, :, :, 1])
    ifftn_mpi!(parallel, K, plan, j.*(K[3].*input[:, :, :, 1] .- K[1].*input[:, :, :, 3]), out[:, :, :, 2])
    ifftn_mpi!(parallel, K, plan, j.*(K[1].*input[:, :, :, 2] .- K[2].*input[:, :, :, 1]), out[:, :, :, 3])

    nothing
end

function curl!(
    parallel::P, 
    K::Wavenumbers, 
    plan::Plan,
    input::Array{ComplexF64, 4}
    ;
    out::Array{Float64, 4}
) where P <: AbstractParallel
    __curl!(parallel, K, plan, input, out = out);
end

function curl!(
    parallel::P, 
    K::WavenumbersGPU,
    plan::Plan,
    input::CuArray{ComplexF64, 4}
    ;
    out::CuArray{Float64, 4}
) where P <: AbstractParallel
    __curl!(parallel, K, plan, input, out = out);
end

# take the cross product of X-Space a,b vectors 
# stores the result in (fourier space) `out` and then return the
@views function __cross!(
    parallel::P, 
    plan::Plan,
    a::XARRAY,
    b::XARRAY
    ;
    out::FARRAY
) where P <: AbstractParallel where FARRAY <: AbstractArray{ComplexF64, 4} where XARRAY <: AbstractArray{Float64, 4}
    fftn_mpi!(parallel, plan, a[:, :, :, 2].*b[:, :, :, 3] .- a[:, :, :, 3].*b[:, :, :, 2], out[:, :, :, 1])
    fftn_mpi!(parallel, plan, a[:, :, :, 3].*b[:, :, :, 1] .- a[:, :, :, 1].*b[:, :, :, 3], out[:, :, :, 2])
    fftn_mpi!(parallel, plan, a[:, :, :, 1].*b[:, :, :, 2] .- a[:, :, :, 2].*b[:, :, :, 1], out[:, :, :, 3])

    nothing
end

function cross!(
    parallel::P, 
    plan::Plan,
    a::Array{Float64, 4},
    b::Array{Float64, 4},
    ;
    out::Array{ComplexF64, 4}
) where P <: AbstractParallel
    __cross!(parallel, plan, a, b, out = out);

    nothing
end

function cross!(
    parallel::P,
    plan::Plan,
    a::CuArray{Float64, 4},
    b::CuArray{Float64, 4},
    ;
    out::CuArray{ComplexF64, 4}
) where P <: AbstractParallel
    __cross!(parallel, plan, a, b, out = out);
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
    state::StateGPU,
    forcing::FORCING
) where P <: AbstractParallel where FORCING <: AbstractForcing
    __compute_rhs!(rk_step, parallel, K, U, U_hat, state, forcing)
end

function compute_rhs!(
    rk_step::Int,
    parallel::P,
    K::Wavenumbers,
    U::Array{Float64, 4},
    U_hat::Array{ComplexF64, 4},
    state::State,
    forcing::FORCING
) where P <: AbstractParallel where FORCING <: AbstractForcing
    __compute_rhs!(rk_step, parallel, K, U, U_hat, state, forcing)
end

function __compute_rhs!(
    rk_step::Int,
    parallel::P,
    K::WAVE,
    U::ARRAY,
    U_hat::FARRAY,
    state::STATE,
    forcing::FORCING
) where P <: AbstractParallel where FARRAY <: AbstractArray{ComplexF64, 4} where ARRAY <: AbstractArray{Float64, 4} where STATE <: AbstractState where WAVE <: AbstractWavenumbers where FORCING <: AbstractForcing

    if rk_step != 1
        @views for i in 1:3
            ifftn_mpi!(parallel, K, state.fft_plan, U_hat[:, :, :, i], U[:, :, :, i])
        end
    end

    curl!(parallel, K, state.fft_plan, U_hat; out = state.curl)
    cross!(parallel, state.fft_plan, U, state.curl; out = state.dU)
    state.dU .*= state.dealias

    # compute P_hat = sum(dU * K_over_K²):
    sum!(state.P_hat, state.dU .* state.K_over_K²)

    # compute forcing
    forcing_term = force_system!(parallel, forcing, U_hat, U)
    if forcing_term != nothing
        state.P_hat .+= dropdims(sum(forcing_term .* state.K_over_K²; dims=4); dims=4)
    end
    
    # add Pressure term to dU/dt
    wavenumber_product!(state.P_hat, K; out = state.wavenumber_product_tmp)
    state.dU .-= state.wavenumber_product_tmp

    # now, add the forcing term to the NS RHS
    if forcing_term != nothing
        state.dU .+= forcing_term
    end

    # dU is now the Eulerian term / Nonlinear term in the NSE
    # now calculate the diffusion term
    state.dU .-= state.ν .* state.K² .* U_hat
end

#
end
