include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, Configuration, solver, Integrate, Forcing

using CUDA
using BenchmarkTools

#########
######### GPU benchmark
#########

N = 64
begin
    local parallel = markers.SingleThreadGPU()

    local K = mesh.wavenumbers_gpu(N)
    local cfg = Configuration.taylor_green_validation()
    local msh = mesh.new_mesh(N)
    local ic = markers.TaylorGreen()

    local U = CuArray(zeros(N, N, N, 3))
    local U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    local plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat,plan)
    local st = state.create_state_gpu(N, K, cfg,plan)

    local forcing = Forcing.Unforced()
    local exports = Vector{markers.AbstractIoExport}()

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
    local parallel = markers.SingleThreadCPU()

    local K = mesh.wavenumbers(N)
    local cfg = Configuration.taylor_green_validation()
    local msh = mesh.new_mesh(N)
    local ic = markers.TaylorGreen()

    local U = zeros(N, N, N, 3)
    local U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    local plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)
    local st = state.create_state(N, K, cfg, plan)

    local forcing = Forcing.Unforced()
    local exports = Vector{markers.AbstractIoExport}()

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
