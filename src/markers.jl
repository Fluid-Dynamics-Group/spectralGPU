module markers

#
# Parallization (MPI, Single thread)
#
export AbstractParallel, ParallelMpi, SingleThread

abstract type AbstractParallel end
struct ParallelMpi <: AbstractParallel end
struct SingleThread <: AbstractParallel end

#
# Initial conditions
#
export AbstractInitialCondition, TaylorGreen, LoadInitialCondition

abstract type AbstractInitialCondition end
struct TaylorGreen <: AbstractInitialCondition end
struct LoadInitialCondition <: AbstractInitialCondition end

#
end
