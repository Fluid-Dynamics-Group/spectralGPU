include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft
using Test

@testset "mesh.jl" begin
    # ensure this compiles
    k = mesh.wavenumbers(32)

    @test k.kx[:, 1, 1] == k.kx[:, 2, 2]
    @test k.ky[1, :, 1] == k.ky[2, :, 2]
    @test k.kz[1, 1, :] == k.kz[2, 2, :]
    # Write your tests here.
end
