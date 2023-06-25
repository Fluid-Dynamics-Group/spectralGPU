module Fft

export fftn_mpi!, ifftn_mpi!, Plan, plan_ffts

using FFTW
using CUDA.CUFFT
using CUDA
using LinearAlgebra: mul!

using ..Markers: AbstractParallel, ParallelMpi, SingleThreadCPU, SingleThreadGPU
using ..Mesh: Wavenumbers, WavenumbersGPU

#
# FFT Planning
#

struct Plan
    forward
    inverse
end


function plan_ffts(parallel::SingleThreadCPU, K::Wavenumbers, u::XARR, uhat::FARR)::Plan where XARR <: AbstractArray{Float64, 3} where FARR <: AbstractArray{ComplexF64, 3}
    forward = FFTW.plan_rfft(u, 1:3)
    backward = FFTW.plan_irfft(uhat, K.n, 1:3)
    return Plan(forward, backward)
end

function plan_ffts(parallel::SingleThreadGPU, K::WavenumbersGPU, u::CuArray{Float64,3}, uhat::CuArray{ComplexF64, 3})
    forward = CUFFT.plan_rfft(u, 1:3)
    backward = CUFFT.plan_irfft(uhat, K.n, 1:3)
    return Plan(forward, backward)
end

#
# Forward FFTs
#

fftn_mpi!(parallel::P, u, uhat) where P <: AbstractParallel = error("function is not not yet implemented for $(typeof(uhat)), $(typeof(u)). This is a bug")

# single threaded forward FFT implementation for Base arrays
function fftn_mpi!(parallel::SingleThreadCPU, plan::Plan, u::XARR, uhat::SubArray{ComplexF64, 3}) where XARR <: AbstractArray{Float64, 3}
    # rfft
    #mul!(uhat, plan.forward, u)

    uhat[:, :, :] .= rfft(u, 1:3)

    nothing
end

# single threaded forward FFT implementation for CUDA arrays
function fftn_mpi!(parallel::SingleThreadGPU, plan::Plan, u::CuArray{Float64, 3}, uhat::CuArray{ComplexF64, 3})
    # rfft
    mul!(uhat, plan.forward, u)

    nothing
end

#
# Inverse FFTs
#

ifftn_mpi!(parallel::P, wavenumbers::Wavenumbers, uhat, u) where P <: AbstractParallel = error("function is not not yet implemented for $(typeof(uhat)), $(typeof(u)). This is a bug")

# single threaded inverse FFT implementation for Base arrays
function ifftn_mpi!(parallel::SingleThreadCPU, wavenumbers::Wavenumbers, plan::Plan, uhat::FARR, u::SubArray{Float64, 3}
) where FARR <: AbstractArray{ComplexF64, 3}
    # irfft

    # zero allocation multiplication here causes NaNs in solver, unsure why
    # mul!(u, plan.inverse, uhat)
    #
    # u[:, :, :] .= plan.inverse * uhat

    u[:, :, :] = FFTW.irfft(uhat, wavenumbers.n, 1:3)

    nothing
end

# single threaded inverse FFT implementation for CUDA arrays
function ifftn_mpi!(parallel::SingleThreadGPU, wavenumbers::WavenumbersGPU, plan::Plan, uhat::CuArray{ComplexF64, 3}, u::CuArray{Float64, 3}
)
    # irfft
    mul!(u, plan.inverse, uhat)
    #u[:, :, :] .= irfft(uhat, K.n, 1:3)
    
    nothing
end

#
end
