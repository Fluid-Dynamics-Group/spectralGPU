module mesh

export Wavenumbers, wavenumbers
export Mesh, new_mesh 

using LazyGrids

struct Wavenumbers
    # number of x space values in each direction
    n::Int
    # number of modes in the x direction (N //2) +1
    # so for N = 64, this is 33
    kn::Int
    kx::LazyGrids.GridAV{Int64, 1, 3}
    ky::LazyGrids.GridAV{Int64, 2, 3}
    kz::LazyGrids.GridAV{Int64, 3, 3}
end

function wavenumbers(n::Int)::Wavenumbers
    increasing_wavenumbers_positive = 0:(div(n,2)-1)
    increasing_wavenumbers_negative = -(div(n,2)):-1

    # fft frequencies replicatese numpy fft.fftfreq(N, 1/N)
    ky_linear = vcat([increasing_wavenumbers_positive, increasing_wavenumbers_negative]...)

    # wave numbers in the z direction is identical to y direction
    kz_linear = copy(ky_linear)

    # FFTW abbreviates the data in the x direction
    # div(3,2) = 1, it is integer division
    kn = div(n,2) + 1
    kx_linear = ky_linear[1:kn]

    kx, ky, kz = ndgrid(kx_linear, ky_linear, kz_linear)

    Wavenumbers(n, kn, kx, ky, kz)
end

# make the Wavenumber type indexable
function Base.getindex(k_all::Wavenumbers, dim::Int)::LazyGrids.GridAV
    if dim == 1
        return k_all.kx
    elseif dim == 2
        return k_all.ky
    elseif dim == 3
        return k_all.kz
    else
        error("dimension was not 1,2,3")
    end
end

struct Mesh
    n::Int
    x::LazyGrids.GridSL{Float64, 1, 3, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
    y::LazyGrids.GridSL{Float64, 2, 3, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
    z::LazyGrids.GridSL{Float64, 3, 3, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}
end

# nprange is range but for spectral things. It operates with the same spacing
# parameters as numpy's mgrid[]
# 
# nprange(0, 1, 10) = 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# range(0, 1, 10) = 0.0 0.111111 0.222222 0.333333 0.444444 0.555556 0.666667 0.777778 0.888889 1.0
function nprange(start, _end, n)
    step = (_end - start) / n
    start:step:(_end - step)
end

# create a mesh that is identical to numpy's mgrid[] spacing
function new_mesh(n)
    xvals = nprange(0, 2pi, n)
    yvals = copy(xvals)
    zvals = copy(xvals)

    mesh_x, mesh_y, mesh_z = ndgrid(xvals, yvals, zvals); size(mesh_x)

    return Mesh(n, mesh_x, mesh_y, mesh_z)
end

#
end
