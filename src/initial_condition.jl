module initial_condition

using ..mesh: Mesh
using ..fft: fftn_mpi!, Plan
using ..markers: AbstractInitialCondition, AbstractParallel, TaylorGreen, ABC

using CUDA

# catch-all for initial condition setup
function setup_initial_condition(
    parallel::P, 
    ic::I, 
    mesh::Mesh, 
    U::XARR,
    U_hat::FARR,
) where P <: AbstractParallel where I <: AbstractInitialCondition where XARR <: AbstractArray{Float64, 4} where FARR <: AbstractArray{ComplexF64, 4}
    error("unimplemented initial condition for type " * typeof(ic))
end

# Taylor-Green
function setup_initial_condition(
    parallel::P, 
    ic::TaylorGreen, 
    mesh::Mesh, 
    U::XARR,
    U_hat::FARR,
    plan::Plan
) where P <: AbstractParallel where XARR <: AbstractArray{Float64, 4} where FARR <: AbstractArray{ComplexF64, 4}
    U[:, :, :, 1] .= sin.(mesh.x) .* cos.(mesh.y) .* cos.(mesh.z)
    U[:, :, :, 2] .= cos.(mesh.x) .* sin.(mesh.y) .* cos.(mesh.z)
    U[:, :, :, 2] *= -1
    U[:, :, :, 3] .= 0.

    @views for i in 1:3
        fftn_mpi!(parallel, plan, U[:, :, :, i], U_hat[:, :, :, i])
    end
end

# ABC flow
# https://en.wikipedia.org/wiki/Arnold%E2%80%93Beltrami%E2%80%93Childress_flow
function setup_initial_condition(
    parallel::P, 
    ic::ABC, 
    mesh::Mesh, 
    U::XARR,
    U_hat::FARR,
    plan::Plan
) where P <: AbstractParallel where XARR <: AbstractArray{Float64, 4} where FARR <: AbstractArray{ComplexF64, 4}
    A = 1
    B = 2
    C = 3

    U[:, :, :, 1] .= A .* sin.(mesh.z) .+ C .* cos.(mesh.y)
    U[:, :, :, 2] .= B .* sin.(mesh.x) .+ A .* cos.(mesh.z)
    U[:, :, :, 3] .= C .* sin.(mesh.y) .+ B .* cos.(mesh.x)

    @views for i in 1:3
        fftn_mpi!(parallel, plan, U[:, :, :, i], U_hat[:, :, :, i])
    end
end

end
