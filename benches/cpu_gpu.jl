include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, config, solver, Integrate

using CUDA
using BenchmarkTools

#########
######### GPU benchmark
#########

N = 128
begin
    local parallel = markers.SingleThreadGPU()

    local K = mesh.wavenumbers_gpu(N)
    local cfg = config.taylor_green_validation()
    local msh = mesh.new_mesh(N)
    local ic = markers.TaylorGreen()

    local U = CuArray(zeros(N, N, N, 3))
    local U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    local plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat,plan)
    local st = state.create_state_gpu(N, K, cfg,plan)

    r = @benchmark Integrate.integrate(
        $parallel,
        $K,
        $cfg,
        $st,
        $U,
        $U_hat,
    )

    println("GPU full solver integration")
    display(r)
    println("\n");
end

#########
######### CPU benchmark
#########

begin
    local parallel = markers.SingleThreadCPU()

    local K = mesh.wavenumbers(N)
    local cfg = config.taylor_green_validation()
    local msh = mesh.new_mesh(N)
    local ic = markers.TaylorGreen()

    local U = zeros(N, N, N, 3)
    local U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    local plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)
    local st = state.create_state(N, K, cfg, plan)

    r = @benchmark Integrate.integrate(
        $parallel,
        $K,
        $cfg,
        $st,
        $U,
        $U_hat,
    )

    println("CPU full solver integration")
    display(r)
end
