module fft

export fftn_mpi!, ifftn_mpi!

using FFTW
using ..markers: AbstractParallel, ParallelMpi, SingleThread
using ..mesh: Wavenumbers

fftn_mpi!(parallel::P, u, uhat) where P <: AbstractParallel = error("function is not not yet implemented. This is a bug")

# single threaded forward FFT implementation for Base arrays
function fftn_mpi!(parallel::SingleThread, u::Array{Float64, 3}, uhat::Array{ComplexF64, 3}) 
    uhat[:, :, :] .= rfft(u, 1:3)
end

ifftn_mpi!(parallel::P, wavenumbers::Wavenumbers, uhat, u) where P <: AbstractParallel = error("function is not not yet implemented. This is a bug")

# single threaded inverse FFT implementation for Base arrays
function ifftn_mpi!(parallel::SingleThread, wavenumbers::Wavenumbers, uhat::Array{ComplexF64, 3}, u::Array{Float64, 3})
    u[:, :, :] .= irfft(uhat, wavenumbers.n, 1:3)
end

#
end
