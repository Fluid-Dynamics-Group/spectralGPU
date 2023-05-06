include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, config, solver
using Test

@testset "mesh.jl" begin
    # ensure this compiles
    k = mesh.wavenumbers(32)

    @test k.kx[:, 1, 1] == k.kx[:, 2, 2]
    @test k.ky[1, :, 1] == k.ky[2, :, 2]
    @test k.kz[1, 1, :] == k.kz[2, 2, :]
end

@testset "initial_condition.jl" begin
    parallel = markers.SingleThread()
    N = 64

    K = mesh.wavenumbers(N)
    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))

    msh = mesh.new_mesh(N)

    # taylor green
    @test begin
        ic = markers.TaylorGreen()
        initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat)
        true
    end
end

@testset "solver.jl" begin
    parallel = markers.SingleThread()
    N = 64
    re = 40.

    K = mesh.wavenumbers(N)
    st = state.create_state(N, K)
    msh = mesh.new_mesh(N)
    cfg = config.create_config(N, re)

    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))

    # curl
    @test begin
        #solver.curl!(K, U_hat; out = st.curl[:, :, :, :])
        solver.curl!(parallel, K, U_hat; out=st.curl)
        true
    end

    # cross
    @test begin
        solver.cross!(parallel, U, st.curl; out = st.dU)
        true
    end

    #solver.compute_rhs!(
    #    0,
    #    parallel,
    #    K,
    #    cfg,
    #    U,
    #    U_hat,
    #    st
    #)
end
