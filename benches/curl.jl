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
    local st = state.create_state_gpu(N, K, cfg)
    local U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))

    r = @benchmark solver.curl!($parallel, $K, $U_hat; out = $st.curl)

    println("GPU curl")
    display(r)
    println("")
end

begin
    local parallel = markers.SingleThreadCPU()
    local re = 40.
    local K = mesh.wavenumbers(N)
    local cfg = config.create_config(N, re, 0.00001)
    local st = state.create_state(N, K, cfg)
    local U_hat = ComplexF64.(zeros(K.kn, N, N, 3))

    r = @benchmark solver.curl!($parallel, $K, $U_hat; out = $st.curl)

    println("CPU curl")
    display(r)
    println("")
end
