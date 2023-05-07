module fft

export fftn_mpi!, ifftn_mpi!

using FFTW
using ..markers: AbstractParallel, ParallelMpi, SingleThread
using ..mesh: Wavenumbers

fftn_mpi!(parallel::P, u, uhat) where P <: AbstractParallel = error("function is not not yet implemented for $(typeof(uhat)), $(typeof(u)). This is a bug")

# single threaded forward FFT implementation for Base arrays
function fftn_mpi!(parallel::SingleThread, u::XARR, uhat::SubArray{ComplexF64, 3}) where XARR <: AbstractArray{Float64, 3}
    uhat[:, :, :] .= rfft(u, 1:3)

    nothing
end

ifftn_mpi!(parallel::P, wavenumbers::Wavenumbers, uhat, u) where P <: AbstractParallel = error("function is not not yet implemented for $(typeof(uhat)), $(typeof(u)). This is a bug")

# single threaded inverse FFT implementation for Base arrays
function ifftn_mpi!(parallel::SingleThread, wavenumbers::Wavenumbers, uhat::FARR, u::SubArray{Float64, 3}
) where FARR <: AbstractArray{ComplexF64, 3}
    u[:, :, :] .= irfft(uhat, wavenumbers.n, 1:3)
end

#
end
