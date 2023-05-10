module state

using ..mesh: Wavenumbers
using ..markers: AbstractState, AbstractWavenumbers, AbstractConfig
using ..config: Config

using CUDA

export State, create_state, create_state_gpu

struct State <: AbstractState
    dU::Array{ComplexF64, 4}
    U_hat₁::Array{ComplexF64, 4}
    U_hat₀::Array{ComplexF64, 4}
    curl::Array{Float64, 4}
    dealias::BitArray{3}
    P_hat::Array{ComplexF64, 3}
    K²::Array{Int, 3}
    K_over_K²::Array{Float64, 4}
    wavenumber_product_tmp::Array{ComplexF64, 4}
    ν::Float64
    a::Vector{Float64}
    b::Vector{Float64}
end

struct StateGPU <: AbstractState
    dU::CuArray{ComplexF64, 4}
    U_hat₁::CuArray{ComplexF64, 4}
    U_hat₀::CuArray{ComplexF64, 4}
    curl::CuArray{Float64, 4}
    dealias::CuArray{Int8, 3}
    P_hat::CuArray{ComplexF64, 3}
    K²::CuArray{Int, 3}
    K_over_K²::CuArray{Float64, 4}
    wavenumber_product_tmp::CuArray{ComplexF64, 4}
    ν::CuVector{Float64}
    # RK integration constants
    a::Vector{Float64}
    # RK integration constants
    b::Vector{Float64}
end

function create_state(N::Int, K::WAVE, config::Config{CFG})::State where WAVE <: AbstractWavenumbers where CFG <: AbstractConfig
    dU = ComplexF64.(zeros(K.kn, N, N, 3))
    U_hat₁ = ComplexF64.(zeros(K.kn, N, N, 3))
    U_hat₀ = ComplexF64.(zeros(K.kn, N, N, 3))

    curl = zeros(N, N, N, 3)
    P_hat = ComplexF64.(zeros(K.kn, N, N))
    wavenumber_product_tmp = ComplexF64.(zeros(K.kn, N, N, 3))

    K² = Array(K[1].^2 + K[2].^2 + K[3].^2)

    # create a K² array that does not have any zeros
    # this is for the K/K² calculation that follows. If this 
    # term has any zeros then it will produce NaNs in the division
    K²_nonzero = Array(copy(K²))
    K²_nonzero[K² .== 0] .= 1

    kmax_dealias = 2/3 * K.kn
    dealias = (abs.(Array(K.kx)) .< kmax_dealias) .* (abs.(Array(K.ky)) .< kmax_dealias) .* (abs.(Array(K.kz)) .< kmax_dealias)

    K_over_K² = Float64.(zeros(K.kn, N, N, 3))
    for i in 1:3
        K_over_K²[:, :, :, i] .= Float64.(Array(K[i]))
    end

    K_over_K² ./= K²_nonzero

    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Examples
    # a_j is equal to the canonical b_i
    # b_i is equal to the canonical a_j
    a = [1/6, 1/3, 1/3, 1/6]
    b = [0.5, 0.5, 1.]

    State(
        dU,
        U_hat₁,
        U_hat₀,
        curl,
        dealias,
        P_hat,
        K²,
        K_over_K²,
        wavenumber_product_tmp,
        config.ν,
        a,
        b
    )
end

function create_state_gpu(N::Int, K::WAVE, config::Config{CFG})::StateGPU where WAVE <: AbstractWavenumbers where CFG <: AbstractConfig
    cpu_state = create_state(N, K, config)
    dealias::Array{Int8, 3} = Array(cpu_state.dealias)

    return StateGPU(
        #
        CuArray(cpu_state.dU),
        CuArray(cpu_state.U_hat₁),
        CuArray(cpu_state.U_hat₀),
        CuArray(cpu_state.curl),
        CuArray(dealias),
        CuArray(cpu_state.P_hat),
        CuArray(cpu_state.K²),
        CuArray(cpu_state.K_over_K²),
        CuArray(cpu_state.wavenumber_product_tmp),
        CuVector([cpu_state.ν]),
        #CuVector(cpu_state.a),
        #CuVector(cpu_state.b),
        cpu_state.a,
        cpu_state.b,
    )
end

#
end
