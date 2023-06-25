include("../src/spectralGPU.jl");
using .spectralGPU: Mesh, Fft, Markers, InitialCondition, State, Configuration, Solver, Integrate, Forcing

using CUDA
using BenchmarkTools

#########
######### GPU benchmark
#########

N = 64
begin
    local parallel = Markers.SingleThreadGPU()

    local K = Mesh.wavenumbers_gpu(N)
    local cfg = Configuration.taylor_green_validation()
    local msh = Mesh.new_mesh(N)
    local ic = InitialCondition.TaylorGreen()

    local U = CuArray(zeros(N, N, N, 3))
    local U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    local plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    InitialCondition.setup_initial_condition(parallel, ic, msh, U, U_hat,plan)
    local st = State.create_state_gpu(N, K, cfg,plan)

    local forcing = Forcing.Unforced()
    local exports = Vector{Markers.AbstractIoExport}()

    r = @benchmark Integrate.integrate(
        $parallel,
        $K,
        $cfg,
        $st,
        $U,
        $U_hat,
        $forcing,
        $exports
    )

    println("GPU full solver integration")
    display(r)
    println("\n");
end

#########
######### CPU benchmark
#########

begin
    local parallel = Markers.SingleThreadCPU()

    local K = Mesh.wavenumbers(N)
    local cfg = Configuration.taylor_green_validation()
    local msh = Mesh.new_mesh(N)
    local ic = InitialCondition.TaylorGreen()

    local U = zeros(N, N, N, 3)
    local U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    local plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    InitialCondition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)
    local st = State.create_state(N, K, cfg, plan)

    local forcing = Forcing.Unforced()
    local exports = Vector{Markers.AbstractIoExport}()

    r = @benchmark Integrate.integrate(
        $parallel,
        $K,
        $cfg,
        $st,
        $U,
        $U_hat,
        $forcing,
        $exports
    )

    println("CPU full solver integration")
    display(r)
end
