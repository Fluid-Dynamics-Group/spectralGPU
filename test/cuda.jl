include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, config, solver, Integrate
using Test
using CUDA


@testset "cuda fft" begin
    parallel = markers.SingleThreadGPU()
    N = 64

    K = mesh.wavenumbers(N)

    U = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))

    @test begin
        @views fft.fftn_mpi!(parallel, U[:, :, :, 1], U_hat[:, :, :, 1])
        true
    end

    @test begin
        @views fft.ifftn_mpi!(parallel, K, U_hat[:, :, :, 1], U[:, :, :, 1])
        true
    end
end


@testset "cuda fft initial condition" begin
    parallel = markers.SingleThreadGPU()
    N = 64

    K = mesh.wavenumbers(N)

    U = CuArray(zeros(N, N, N, 3))
    U_inverse = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))

    ic = markers.TaylorGreen()
    msh = mesh.new_mesh(N)

    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat)

    u_hat_sum = sum(abs.(U_hat))

    # ensure the velocity component has been populated
    @test begin
        U_norm = sum(abs.(U))
        U_norm > 1000
    end

    # ensure forward FFT works well for CUDA arrays
    @test begin
        if u_hat_sum == 0
            println("uhat sum was zero: ", u_hat_sum)
            false
        else
            true
        end
    end

    # ensure the inverse FFT works equally well for CUDA arrays
    @test begin
        @views for i in 1:3
            fft.ifftn_mpi!(parallel, K, U_hat[:, :, :, i], U_inverse[:, :, :, i])
        end

        diff = U - U_inverse
        l1_error = sum(abs.(diff))
        l1_error < 1e-5
    end
end

#@testset "solver.jl" begin
#    parallel = markers.SingleThreadGPU()
#    N = 64
#    re = 40.
#
#    K = mesh.wavenumbers(N)
#    st = state.create_state(N, K)
#    msh = mesh.new_mesh(N)
#    cfg = config.create_config(N, re, 1.0)
#
#    U = CuArray(zeros(N, N, N, 3))
#    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
#
#    # curl
#    @test begin
#        solver.curl!(parallel, K, U_hat; out=st.curl)
#        true
#    end
#
#    # cross
#    @test begin
#        solver.cross!(parallel, U, st.curl; out = st.dU)
#        true
#    end
#
#    # main solver call
#    @test begin
#        solver.compute_rhs!(
#            2,
#            parallel,
#            K,
#            cfg,
#            U,
#            U_hat,
#            st
#        )
#        true
#    end
#
#end
