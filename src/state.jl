module state

using ..mesh: Wavenumbers

export State, create_state

struct State
    dU::Array{ComplexF64, 4}
    U_hat₁::Array{ComplexF64, 4}
    U_hat₀::Array{ComplexF64, 4}
    curl::Array{Float64, 4}
    dealias::BitArray{3}
    P_hat::Array{ComplexF64, 3}
    K²::Array{Int, 3}
    K_over_K²::Array{Float64, 4}
    wavenumber_product_tmp::Array{ComplexF64, 4}
end

function create_state(N::Int, K::Wavenumbers)::State
    dU = ComplexF64.(zeros(K.kn, N, N, 3))
    U_hat₁ = ComplexF64.(zeros(K.kn, N, N, 3))
    U_hat₀ = ComplexF64.(zeros(K.kn, N, N, 3))

    curl = zeros(N, N, N, 3)
    P_hat = ComplexF64.(zeros(K.kn, N, N))
    wavenumber_product_tmp = ComplexF64.(zeros(K.kn, N, N, 3))

    K² = K[1].^2 + K[2].^2 + K[3].^2

    # create a K² array that does not have any zeros
    # this is for the K/K² calculation that follows. If this 
    # term has any zeros then it will produce NaNs in the division
    K²_nonzero = copy(K²)
    K²_nonzero[K² .== 0] .= 1

    kmax_dealias = 2/3 * K.kn
    dealias = (abs.(K.kx) .< kmax_dealias) .* (abs.(K.ky) .< kmax_dealias) .* (abs.(K.kz) .< kmax_dealias)

    K_over_K² = Float64.(zeros(K.kn, N, N, 3))
    for i in 1:3
        K_over_K²[:, :, :, i] .= Float64.(K[i])
    end

    K_over_K² ./= K²_nonzero

    State(
        dU,
        U_hat₁,
        U_hat₀,
        curl,
        dealias,
        P_hat,
        K²,
        K_over_K²,
        wavenumber_product_tmp
    )
end

#
end
