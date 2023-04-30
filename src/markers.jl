module markers

export AbstractParallel, ParallelMpi, SingleThread

abstract type AbstractParallel end
struct ParallelMpi <: AbstractParallel end
struct SingleThread <: AbstractParallel end


#
end
