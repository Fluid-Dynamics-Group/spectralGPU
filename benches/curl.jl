include("../src/spectralGPU.jl");
using .spectralGPU: Mesh, Fft, Markers, State, Configuration, Solver

using CUDA
using BenchmarkTools

N = 128

begin
    local parallel = Markers.SingleThreadGPU()
    local re = 40.
    local K = Mesh.wavenumbers_gpu(N)
    local U = CuArray(zeros(K.kn, N, N, 3))
    local U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    local cfg = Configuration.create_config(N, re, 0.00001, U)
    local plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    local st = State.create_state_gpu(N, K, cfg, plan)

    r = @benchmark Solver.curl!($parallel, $K, $plan, $U_hat; out = $st.curl)

    println("GPU curl")
    display(r)
    println("")
end

begin
    local parallel = Markers.SingleThreadCPU()
    local re = 40.
    local K = Mesh.wavenumbers(N)
    local U = zeros(K.kn, N, N, 3)
    local U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    local cfg = Configuration.create_config(N, re, 0.00001, U)
    local plan = Fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    local st = State.create_state(N, K, cfg, plan)

    r = @benchmark Solver.curl!($parallel, $K, $plan, $U_hat; out = $st.curl)

    println("CPU curl")
    display(r)
    println("")
end
