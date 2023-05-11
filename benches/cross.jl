include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, config, solver, Integrate

using CUDA
using BenchmarkTools

N = 128

begin
    local parallel = markers.SingleThreadGPU()
    local re = 40.
    local K = mesh.wavenumbers_gpu(N)
    local cfg = config.create_config(N, re, 0.00001)
    local U = CuArray(zeros(N, N, N, 3))
    local U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    local plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    local st = state.create_state_gpu(N, K, cfg, plan)

    r = @benchmark solver.cross!($parallel, $plan, $U, $st.curl; out = $st.dU)

    println("GPU cross")
    display(r)
    println("")
end

begin
    local parallel = markers.SingleThreadCPU()
    local re = 40.
    local K = mesh.wavenumbers(N)
    local cfg = config.create_config(N, re, 0.00001)
    local U = zeros(N, N, N, 3)
    local U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    local plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    local st = state.create_state(N, K, cfg, plan)

    r = @benchmark solver.cross!($parallel, $plan, $U, $st.curl; out = $st.dU)

    println("CPU cross")
    display(r)
    println("")
end
