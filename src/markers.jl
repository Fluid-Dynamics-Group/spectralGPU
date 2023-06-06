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
struct ABC <: AbstractInitialCondition end
struct LoadInitialCondition <: AbstractInitialCondition end

# State
export AbstractState

abstract type AbstractState end

# Wavenumbers
export AbstractWavenumbers

abstract type AbstractWavenumbers end

# Config
export AbstractConfig, ProductionConfig, ValidationConfig

abstract type AbstractConfig end
struct ProductionConfig <: AbstractConfig 
    max_velocity::Float64
end
struct ValidationConfig <: AbstractConfig 
    fixed_dt::Float64
end

# forcing
export AbstractForcing

abstract type AbstractForcing end

# IoStepper
export AbstractIoStepControl

abstract type AbstractIoStepControl end

# IoExporter
export AbstractIoExport

abstract type AbstractIoExport end

# Export History
export AbstractHistory, AbstractScalarHistory, AbstractVectorFieldHistory, AbstractScalarFieldHistory

abstract type AbstractHistory end
# TODO: document required interfaces
abstract type AbstractScalarHistory <: AbstractHistory end
# TODO: document required interfaces
abstract type AbstractVectorFieldHistory <: AbstractHistory end
# TODO: document required interfaces
abstract type AbstractScalarFieldHistory <: AbstractHistory end

#
end
