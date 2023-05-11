include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, config, solver, Integrate

using CUDA
using BenchmarkTools

#########
######### GPU (forward) benchmark
#########

N = 128

begin
    parallel = markers.SingleThreadGPU()

    K = mesh.wavenumbers_gpu(N)

    U = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))

    msh = mesh.new_mesh(N)
    ic = markers.TaylorGreen()
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat)

    r = @benchmark @views fft.fftn_mpi!($parallel, $U[:, :, :, 1], $U_hat[:, :, :, 1])

    println("gpu forward FFT")
    display(r)
    println("")
end

#########
######### CPU (forward) benchmark
#########

begin
    parallel = markers.SingleThreadCPU()

    K = mesh.wavenumbers(N)

    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))

    msh = mesh.new_mesh(N)
    ic = markers.TaylorGreen()
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat)

    r = @benchmark @views fft.fftn_mpi!($parallel, $U[:, :, :, 1], $U_hat[:, :, :, 1])

    println("cpu forward FFT")
    display(r)
    println("")
end

#########
######### GPU (inverse) benchmark
#########

begin
    parallel = markers.SingleThreadGPU()

    K = mesh.wavenumbers_gpu(N)

    U = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))

    msh = mesh.new_mesh(N)
    ic = markers.TaylorGreen()
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat)

    r = @benchmark @views fft.ifftn_mpi!($parallel, $K, $U_hat[:, :, :, 1], $U[:, :, :, 1])

    println("gpu inverse FFT")
    display(r)
    println("")
end

#########
######### CPU (inverse) benchmark
#########

begin
    parallel = markers.SingleThreadCPU()

    K = mesh.wavenumbers(N)

    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))

    msh = mesh.new_mesh(N)
    ic = markers.TaylorGreen()
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat)

    r = @benchmark @views fft.ifftn_mpi!($parallel, $K, $U_hat[:, :, :, 1], $U[:, :, :, 1])

    println("cpu inverse FFT")
    display(r)
    println("")
end
