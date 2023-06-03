module Forcing

export Unforced, force_system!

using ..markers: AbstractState, AbstractForcing, AbstractParallel
using Printf

# structure representing an unforced system
struct Unforced <: AbstractForcing end

# required function for the appropriate calculation of forcing. 
# returns either `nothing` (no forcing applied to the system)` or a fourier-space array
function force_system!(
    parallel::P, 
    forcing::FORCING, 
    U_hat::FARRAY, 
    U::XARRAY
)::Union{FARRAY, Nothing} where FORCING <: AbstractForcing where FARRAY <: AbstractArray{ComplexF64, 4} where XARRAY <: AbstractArray{Float64, 4} where P <: AbstractParallel
    error(@sprintf "forcing system %s not defined" typeof(forcing))
end

# blanket function for unforced systems
function force_system!(
    parallel::P,
    forcing::Unforced,
    U_hat::FARRAY,
    U::XARRAY
)::Union{FARRAY, Nothing} where FARRAY <: AbstractArray{ComplexF64, 4} where XARRAY <: AbstractArray{Float64, 4} where P <: AbstractParallel
    # return no forcing, always
    nothing
end

#
end
