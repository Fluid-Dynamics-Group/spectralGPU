module fft

export fftn_mpi!, ifftn_mpi!

using FFTW
using CUDA.CUFFT
using CUDA
using ..markers: AbstractParallel, ParallelMpi, SingleThreadCPU, SingleThreadGPU
using ..mesh: Wavenumbers

fftn_mpi!(parallel::P, u, uhat) where P <: AbstractParallel = error("function is not not yet implemented for $(typeof(uhat)), $(typeof(u)). This is a bug")

# single threaded forward FFT implementation for Base arrays
function fftn_mpi!(parallel::SingleThreadCPU, u::XARR, uhat::SubArray{ComplexF64, 3}) where XARR <: AbstractArray{Float64, 3}
    uhat[:, :, :] .= FFTW.rfft(u, 1:3)

    nothing
end

# single threaded forward FFT implementation for CUDA arrays
function fftn_mpi!(parallel::SingleThreadGPU, u::CuArray{Float64, 3}, uhat::CuArray{ComplexF64, 3})
    uhat[:, :, :] .= CUFFT.rfft(u, 1:3)

    nothing
end

ifftn_mpi!(parallel::P, wavenumbers::Wavenumbers, uhat, u) where P <: AbstractParallel = error("function is not not yet implemented for $(typeof(uhat)), $(typeof(u)). This is a bug")

# single threaded inverse FFT implementation for Base arrays
function ifftn_mpi!(parallel::SingleThreadCPU, wavenumbers::Wavenumbers, uhat::FARR, u::SubArray{Float64, 3}
) where FARR <: AbstractArray{ComplexF64, 3}
    u[:, :, :] .= FFTW.irfft(uhat, wavenumbers.n, 1:3)
end

# single threaded inverse FFT implementation for CUDA arrays
function ifftn_mpi!(parallel::SingleThreadGPU, wavenumbers::Wavenumbers, uhat::CuArray{ComplexF64, 3}, u::CuArray{Float64, 3}
)
    u[:, :, :] .= CUFFT.irfft(uhat, wavenumbers.n, 1:3)
end

#
end
