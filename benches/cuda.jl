include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, config, solver, Integrate
using CUDA
using BenchmarkTools

function multiply_cuda(arr::CuArray{Float64,4}, scalar::CuArray{Float64,1})
    arr .* scalar
end

function multiply_cpu(arr::CuArray{Float64,4}, scalar::Float64)
    arr .* scalar
end

const N = 512

begin
    local scalar = Float64.(CUDA.randn(1))
    r = @benchmark multiply_cuda(arr, $scalar) setup=(arr = Float64.(CUDA.randn(N, N, N, 3)))

    println("cuda scalar multiply, all gpu")
    display(r)
    println("")
end

begin
    local scalar::Float64 = rand()
    r = @benchmark multiply_cpu(arr, $scalar) setup=(arr = Float64.(CUDA.randn(N, N, N, 3)))

    println("cuda scalar multiply, scalar is cpu")
    display(r)
    println("")
end
