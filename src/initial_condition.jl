module initial_condition

using ..mesh: Mesh
using ..fft: fftn_mpi!
using ..markers: AbstractInitialCondition, AbstractParallel, TaylorGreen

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
) where P <: AbstractParallel where XARR <: AbstractArray{Float64, 4} where FARR <: AbstractArray{ComplexF64, 4}
    @CUDA.sync U[:, :, :, 1] .= sin.(mesh.x) .* cos.(mesh.y) .* cos.(mesh.z)
    @CUDA.sync U[:, :, :, 2] .= cos.(mesh.x) .* sin.(mesh.y) .* cos.(mesh.z)
    @CUDA.sync U[:, :, :, 2] *= -1
    @CUDA.sync U[:, :, :, 3] .= 0.

    @views for i in 1:3
        fftn_mpi!(parallel, U[:, :, :, i], U_hat[:, :, :, i])
    end
end

end
