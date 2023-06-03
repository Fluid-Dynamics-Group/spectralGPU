include("../src/spectralGPU.jl");
using .spectralGPU: mesh, fft, markers, initial_condition, state, config, solver, Integrate, Forcing, Io
using Test
using Printf

@testset "mesh.jl" begin
    # ensure this compiles
    k = mesh.wavenumbers(32)

    @test k.kx[:, 1, 1] == k.kx[:, 2, 2]
    @test k.ky[1, :, 1] == k.ky[2, :, 2]
    @test k.kz[1, 1, :] == k.kz[2, 2, :]
end

@testset "io.jl" begin
    @test_throws ErrorException begin
        never = Io.never_write()
        Io.step_number(never)
    end

    @test begin
        dt_write = Io.dt_write(0.1)
        Io.step_number(dt_write) == 1
    end

    #
    # individual timestep checks
    #

    @test begin
        # stepper dt is larger than solver dt, but we want to write the initial timestep
        # regardless
        dt_write = Io.dt_write(0.6)
        solver_dt = 0.5
        current_solver_time = 10.0
        Io.should_write(dt_write, current_solver_time) == true 
    end

    @test begin
        # stepper dt is smaller than solver dt, it should naturally write the first step
        dt_write = Io.dt_write(0.1)
        solver_dt = 0.5
        current_solver_time = 10.0
        Io.should_write(dt_write, current_solver_time) == true 
    end

    @test begin
        # stepper dt is smaller than solver dt, it should also write the second step
        dt_write = Io.dt_write(0.1)

        solver_dt = 0.5
        current_solver_time = 10.0

        Io.increase_step!(dt_write)

        # check for the second step
        Io.should_write(dt_write, current_solver_time) == true 
    end

    #
    # total number of writes checks
    #
    @test begin
        never = Io.never_write()
        Io.num_writes(never, 0.1, 10.0) == 0
    end

    @test begin
        dt_write = Io.dt_write(0.1)
        solver_dt = 0.01
        solver_t = 10.0
        writes = Io.num_writes(dt_write, solver_dt, solver_t)
        Io.num_writes(dt_write, solver_dt, solver_t) == 100
    end

    @test begin
        dt_write = Io.dt_write(0.1)
        solver_dt = 0.5
        solver_t = 10.0
        Io.num_writes(dt_write, solver_dt, solver_t) == 20
    end

    @test begin
        dt_write = Io.dt_write(0.1)
        solver_dt = 0.5
        solver_t = 10.01
        Io.num_writes(dt_write, solver_dt, solver_t) == 21
    end
end

@testset "fft.jl planning" begin
    N = 64
    parallel = markers.SingleThreadCPU();
    K = mesh.wavenumbers(N)
    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))

    @test begin
        fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])
        true
    end
end

@testset "initial_condition.jl" begin
    parallel = markers.SingleThreadCPU()
    N = 64

    K = mesh.wavenumbers(N)
    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))

    msh = mesh.new_mesh(N)

    plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    # taylor green
    @test begin
        ic = markers.TaylorGreen()
        initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)
        u_hat_sum = sum(abs.(U_hat))

        if u_hat_sum == 0
            println("uhat sum was zero: ", u_hat_sum)
            false
        else
            true
        end
    end
end

@testset "solver.jl" begin
    parallel = markers.SingleThreadCPU()
    N = 64
    re = 40.

    K = mesh.wavenumbers(N)

    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))

    cfg = config.taylor_green_validation()
    plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    st = state.create_state(N, K, cfg, plan)
    msh = mesh.new_mesh(N)

    forcing = Forcing.Unforced();

    # curl
    @test begin
        #solver.curl!(K, U_hat; out = st.curl[:, :, :, :])
        solver.curl!(parallel, K, st.fft_plan, U_hat; out=st.curl)
        true
    end

    # cross
    @test begin
        solver.cross!(parallel, st.fft_plan, U, st.curl; out = st.dU)
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
            st,
            forcing
        )
        true
    end

    ########
    ######## Now, begin numerical validation of curl / cross / RHS
    ########

    initial_condition.setup_initial_condition(parallel, initial_condition.TaylorGreen(), msh, U, U_hat, plan)

    @test begin
        solver.curl!(parallel, K, st.fft_plan, U_hat; out=st.curl)

        result = Int(floor(sum(abs.(st.curl))))
        t1_result = Int(floor(sum(abs.(st.curl[:, :, :, 1]))))
        t2_result = Int(floor(sum(abs.(st.curl[:, :, :, 2]))))
        t3_result = Int(floor(sum(abs.(st.curl[:, :, :, 3]))))

        total_magnitude = 269893 
        term_1 = 67473
        term_2 = 67473
        term_3 = 134946
        match = (result == total_magnitude) &&
            (t1_result == term_1) &&
            (t2_result == term_2) &&
            (t3_result == term_3)

        if !match
            println("failed match curl")
            println("magnitude was $result - goal: $total_magnitude")

            println("term 1 was $t1_result - goal $term_1")
            println("term 2 was $t2_result - goal $term_2")
            println("term 3 was $t3_result - goal $term_3")
        end

        match
    end

    @test begin
        cross = zeros(N, N, N, 3)
        solver.cross!(parallel, st.fft_plan, U, st.curl; out = st.dU)

        # inverse FFT the data to X space for comparissons
        @views for i in 1:3
            fft.ifftn_mpi!(
                parallel, K, plan, 
                st.dU[:, :, :, i], cross[:, :, :, i]
            )
        end
        result = Int(floor(sum(abs.(cross))))
        t1_result = Int(floor(sum(abs.(cross[:, :, :, 1]))))
        t2_result = Int(floor(sum(abs.(cross[:, :, :, 2]))))
        t3_result = Int(floor(sum(abs.(cross[:, :, :, 3]))))

        total_magnitude = 124762
        all_terms = 41587
        match = (result == total_magnitude) &&
            (t1_result == all_terms) &&
            (t2_result == all_terms) &&
            (t3_result == all_terms)

        if !match
            println("failed match cross")
            println("magnitude was $result - goal: $total_magnitude")

            println("term 1 was $t1_result - goal $all_terms div: $(t1_result/all_terms)")
            println("term 2 was $t2_result - goal $all_terms div: $(t2_result/all_terms)")
            println("term 3 was $t3_result - goal $all_terms div: $(t3_result/all_terms)")
        end

        match
    end

    # main solver call
    @test begin
        dU_real = zeros(N, N, N, 3)
        solver.compute_rhs!(
            1,
            parallel,
            K,
            U,
            U_hat,
            st,
            forcing
        )

        # inverse FFT the data to X space for comparissons
        @views for i in 1:3
            fft.ifftn_mpi!(
                parallel, K, plan, 
                st.dU[:, :, :, i], dU_real[:, :, :, i]
            )
        end

        result = Int(floor(sum(abs.(dU_real))))
        t1_result = Int(floor(sum(abs.(dU_real[:, :, :, 1]))))
        t2_result = Int(floor(sum(abs.(dU_real[:, :, :, 2]))))
        t3_result = Int(floor(sum(abs.(dU_real[:, :, :, 3]))))

        total_magnitude = 43247
        t1_t2 = 13209
        t3 = 16827

        match = (result == total_magnitude) &&
            (t1_result == t1_t2) &&
            (t2_result == t1_t2) &&
            (t3_result == t3)

        if !match
            println("failed match compute RHS")
            println("magnitude was $result - goal: $total_magnitude")

            println("term 1 was $t1_result - goal $t1_t2 div: $(t1_result/t1_t2)")
            println("term 2 was $t2_result - goal $t1_t2 div: $(t2_result/t1_t2)")
            println("term 3 was $t3_result - goal $t3 div: $(t3_result/t3)")
        end

        match
    end

end

@testset "integrate.jl" begin
    parallel = markers.SingleThreadCPU()
    N = 64
    re = 40.
    time = 0.05

    K = mesh.wavenumbers(N)
    cfg = config.create_config(N, re, time)
    msh = mesh.new_mesh(N)

    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    st = state.create_state(N, K, cfg, plan)

    forcing = Forcing.Unforced();

    # main solver call
    @test begin
        Integrate.integrate(
            parallel,
            K,
            cfg,
            st,
            U,
            U_hat,
            forcing,
            Vector{markers.AbstractIoExport}()
        )
        true
    end
end

@testset "integrate.jl - checked" begin
    parallel = markers.SingleThreadCPU()
    N = 64

    forcing = Forcing.Unforced();

    K = mesh.wavenumbers(N)
    cfg = config.taylor_green_validation()
    msh = mesh.new_mesh(N)
    ic = markers.TaylorGreen()

    U = zeros(N, N, N, 3)
    U_hat = ComplexF64.(zeros(K.kn, N, N, 3))
    plan = fft.plan_ffts(parallel, K, U[:, :, :, 1], U_hat[:, :, :, 1])

    initial_condition.setup_initial_condition(parallel, ic, msh, U, U_hat, plan)

    st = state.create_state(N, K, cfg, plan)

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
            forcing,
            Vector{markers.AbstractIoExport}(),
        )

        u_sum = sum(abs.(U))
        println("after some integration, sum of all values in U is ", u_sum);

        k = (1/2) * sum(U .* U) * (1 / N)^3
        println("k is ", k)
        round(k - 0.124953117517; digits=7) == 0
    end
end
