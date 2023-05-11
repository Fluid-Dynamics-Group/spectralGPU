include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, config, solver, Integrate
using Test
using CUDA

@testset "fft.jl planning" begin
    N = 64
    parallel = markers.SingleThreadGPU();
    K = mesh.wavenumbers_gpu(N)
    U = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))

    @test begin
        fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
        true
    end
end

@testset "state.jl" begin
    parallel = markers.SingleThreadGPU()

    N = 64
    K = mesh.wavenumbers_gpu(N)
    cfg = config.create_config(N, 40., 0.00001)

    U = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))
    plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    @test begin
        st::state.StateGPU = state.create_state_gpu(N, K, cfg, plan)
        true
    end
end

@testset "cuda fft" begin
    parallel = markers.SingleThreadGPU()
    N = 64

    K = mesh.wavenumbers_gpu(N)

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

    K = mesh.wavenumbers_gpu(N)

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

@testset "solver.jl" begin
    parallel = markers.SingleThreadGPU()
    N = 64
    re = 40.

    K = mesh.wavenumbers_gpu(N)
    U = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))

    cfg = config.create_config(N, 40., 0.00001)
    plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    st = state.create_state_gpu(N, K, cfg, plan)

    msh = mesh.new_mesh(N)
    cfg = config.create_config(N, re, 1.0)


    # curl
    @test begin
        solver.curl!(parallel, K, U_hat; out=st.curl)
        true
    end

    # cross
    @test begin
        solver.cross!(parallel, U, st.curl; out = st.dU)
        true
    end

    # main solver call
    @test begin
        solver.compute_rhs!(
            2,
            parallel,
            K,
            U,
            U_hat,
            st
        )
        true
    end

end

@testset "integrate.jl - checked" begin
    parallel = markers.SingleThreadGPU()
    N = 64

    K = mesh.wavenumbers_gpu(N)
    U = CuArray(zeros(N, N, N, 3))
    U_hat = CuArray(ComplexF64.(zeros(K.kn, N, N, 3)))

    plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
    msh = mesh.new_mesh(N)
    cfg = config.taylor_green_validation()
    st = state.create_state_gpu(N, K, cfg, plan)
    ic = markers.TaylorGreen()

    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat)

    u_sum = sum(abs.(U))
    println("sum of all values in U is ", u_sum);
    u_hat_sum = sum(abs.(U_hat))
    println("sum of all values in U_hat is ", u_hat_sum);

    # main solver call
    @test begin
        Integrate.integrate(
            parallel,
            K,
            cfg,
            st,
            U,
            U_hat,
        )

        u_sum = sum(abs.(U))
        println("sum of all values in U is ", u_sum);

        k = (1/2) * sum(U .* U) * (1 / N)^3
        println("k is ", k)
        round(k - 0.124953117517; digits=7) == 0
    end
end
