include("../src/spectralGPU.jl");
using .spectralGPU: Mesh, Fft, Markers, InitialCondition, State, Configuration, Solver, Integrate

using CUDA
using BenchmarkTools

#########
######### GPU (forward) benchmark
#########

N = 128

begin
    parallel = Markers.SingleThreadGPU()

    K = Mesh.wavenumbers_gpu(N)

    U = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    msh = Mesh.new_mesh(N)
    ic = InitialCondition.TaylorGreen()
    InitialCondition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)

    r = @benchmark @views Fft.fftn_mpi!($parallel, $plan, $U[:, :, :, 1], $U_hat[:, :, :, 1])

    println("gpu forward FFT")
    display(r)
    println("")
end

#########
######### CPU (forward) benchmark
#########

begin
    parallel = Markers.SingleThreadCPU()

    K = Mesh.wavenumbers(N)

    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    msh = Mesh.new_mesh(N)
    ic = InitialCondition.TaylorGreen()
    InitialCondition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)

    r = @benchmark @views Fft.fftn_mpi!($parallel, $plan, $U[:, :, :, 1], $U_hat[:, :, :, 1])

    println("cpu forward FFT")
    display(r)
    println("")
end

#########
######### GPU (inverse) benchmark
#########

begin
    parallel = Markers.SingleThreadGPU()

    K = Mesh.wavenumbers_gpu(N)

    U = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    msh = Mesh.new_mesh(N)
    ic = InitialCondition.TaylorGreen()
    InitialCondition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)

    r = @benchmark @views Fft.ifftn_mpi!($parallel, $K, $plan, $U_hat[:, :, :, 1], $U[:, :, :, 1])

    println("gpu inverse FFT")
    display(r)
    println("")
end

#########
######### CPU (inverse) benchmark
#########

begin
    parallel = Markers.SingleThreadCPU()

    K = Mesh.wavenumbers(N)

    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    msh = Mesh.new_mesh(N)
    ic = InitialCondition.TaylorGreen()
    InitialCondition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)

    r = @benchmark @views Fft.ifftn_mpi!($parallel, $K, $plan, $U_hat[:, :, :, 1], $U[:, :, :, 1])

    println("cpu inverse FFT")
    display(r)
    println("")
end
