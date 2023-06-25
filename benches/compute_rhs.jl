include("../src/spectralGPU.jl");
using .spectralGPU: Mesh, Fft, Markers, InitialCondition, State, Configuration, Solver, Forcing

using CUDA
using BenchmarkTools

N = 128

begin
    local rk_step = 2
    local parallel = Markers.SingleThreadGPU()
    local K = Mesh.wavenumbers_gpu(N)

    local U = CuArray(zeros(N, N, N, 3))
    local U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    local plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    local msh = Mesh.new_mesh(N)
    local ic = InitialCondition.TaylorGreen()
    local cfg = Configuration.taylor_green_validation()
    InitialCondition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)
    local st = State.create_state_gpu(N, K, cfg,plan)

    forcing = Forcing.Unforced()

    r = @benchmark @views Solver.compute_rhs!($rk_step, $parallel, $K, $U, $U_hat, $st, $forcing)

    println("GPU compute rhs")
    display(r)
    println("")
end

begin
    local rk_step = 2
    local parallel = Markers.SingleThreadCPU()
    local K = Mesh.wavenumbers(N)

    local U = zeros(N, N, N, 3)
    local U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    local plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    local msh = Mesh.new_mesh(N)
    local ic = InitialCondition.TaylorGreen()
    local cfg = Configuration.taylor_green_validation()
    InitialCondition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)
    local st = State.create_state(N, K, cfg,plan)

    forcing = Forcing.Unforced()

    r = @benchmark @views Solver.compute_rhs!($rk_step, $parallel, $K, $U, $U_hat, $st, $forcing)

    println("CPU compute rhs")
    display(r)
    println("")
end
