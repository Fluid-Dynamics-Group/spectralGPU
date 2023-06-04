include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, Configuration, solver, Integrate, Forcing

using CUDA
using BenchmarkTools

N = 128

begin
    local rk_step = 2
    local parallel = markers.SingleThreadGPU()
    local K = mesh.wavenumbers_gpu(N)

    local U = CuArray(zeros(N, N, N, 3))
    local U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    local plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    local msh = mesh.new_mesh(N)
    local ic = markers.TaylorGreen()
    local cfg = Configuration.taylor_green_validation()
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)
    local st = state.create_state_gpu(N, K, cfg,plan)

    forcing = Forcing.Unforced()

    r = @benchmark @views solver.compute_rhs!($rk_step, $parallel, $K, $U, $U_hat, $st, $forcing)

    println("GPU compute rhs")
    display(r)
    println("")
end

begin
    local rk_step = 2
    local parallel = markers.SingleThreadCPU()
    local K = mesh.wavenumbers(N)

    local U = zeros(N, N, N, 3)
    local U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    local plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    local msh = mesh.new_mesh(N)
    local ic = markers.TaylorGreen()
    local cfg = Configuration.taylor_green_validation()
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)
    local st = state.create_state(N, K, cfg,plan)

    forcing = Forcing.Unforced()

    r = @benchmark @views solver.compute_rhs!($rk_step, $parallel, $K, $U, $U_hat, $st, $forcing)

    println("CPU compute rhs")
    display(r)
    println("")
end
