module markers

#
# Parallization (MPI, Single thread)
#
export AbstractParallel, ParallelMpi, SingleThreadCPU

abstract type AbstractParallel end
struct ParallelMpi <: AbstractParallel end
struct SingleThreadCPU <: AbstractParallel end
struct SingleThreadGPU <: AbstractParallel end

#
# Initial conditions
#
export AbstractInitialCondition, TaylorGreen, LoadInitialCondition

abstract type AbstractInitialCondition end
struct TaylorGreen <: AbstractInitialCondition end
struct LoadInitialCondition <: AbstractInitialCondition end

# State
export AbstractState

abstract type AbstractState end

# Wavenumbers
export AbstractWavenumbers

abstract type AbstractWavenumbers end

#
end
