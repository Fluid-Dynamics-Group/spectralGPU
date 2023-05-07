module initial_condition

using ..mesh: Mesh
using ..fft: fftn_mpi!
using ..markers: AbstractInitialCondition, AbstractParallel, TaylorGreen

# catch-all for initial condition setup
function setup_initial_condition(
    parallel::P, 
    ic::I, 
    mesh::Mesh, 
    U::Array{Float64, 4}, 
    U_hat::Array{ComplexF64, 4}
) where P <: AbstractParallel where I <: AbstractInitialCondition
    error("unimplemented initial condition for type " * typeof(ic))
end

# Taylor-Green
function setup_initial_condition(
    parallel::P, 
    ic::TaylorGreen, 
    mesh::Mesh, 
    U::Array{Float64, 4}, 
    U_hat::Array{ComplexF64, 4}
) where P <: AbstractParallel
    U[:, :, :, 1] .= sin.(mesh.x) .* cos.(mesh.y) .* cos.(mesh.z)
    U[:, :, :, 2] .= - cos.(mesh.x) .* sin.(mesh.y) .* cos.(mesh.z)
    U[:, :, :, 3] .= 0.

    @views for i in 1:3
        fftn_mpi!(parallel, U[:, :, :, i], U_hat[:, :, :, i])
    end
end

end
