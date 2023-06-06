module Exporters

export EnergyExport, HelicityExport, ScalarMemoryHistory, ScalarH5History, scalar_h5_history, scalar_memory_history
export VectorFieldExport, vector_field_h5_history

using ..markers: AbstractIoExport, AbstractParallel, AbstractWavenumbers, AbstractScalarHistory, AbstractIoStepControl, AbstractState, AbstractConfig, AbstractVectorFieldHistory
using ..fft: Plan
import ..solver
import ..Io
import ..Configuration

using HDF5

#################
################# In memory / HDF5 history containers
#################

#
# In memory scalar history
#
struct ScalarMemoryHistory <: AbstractScalarHistory
    len::Int
    inner::Vector{Float64}
end
function Base.setindex!(hist::ScalarMemoryHistory, value::Float64, idx::Int)
    if idx <= hist.len
        hist.inner[idx] = value
    end
end

# must be called after initial condition
function scalar_memory_history(
    stepper::STEP, 
    config::Configuration.Config{CONFIG}
)::ScalarMemoryHistory where STEP <: AbstractIoStepControl where CONFIG <: AbstractConfig
    dt = Configuration.calculate_dt(config)
    runtime = config.time
    len = Io.num_writes(stepper,dt, runtime)
    ScalarMemoryHistory(len, zeros(len))
end

#
# HDF5 scalar history 
#
struct ScalarH5History <: AbstractScalarHistory
    len::Int
    inner::HDF5.Dataset
end
function Base.setindex!(hist::ScalarH5History, value::Float64, idx::Int)
    if idx <= hist.len
        hist.inner[idx] = value
    end
end
function scalar_h5_history(stepper::STEP, h5_file, name::String, config::Configuration.Config{CONFIG})::ScalarH5History where STEP <: AbstractIoStepControl where CONFIG <: AbstractConfig
    dt = Configuration.calculate_dt(config)
    len = Io.num_writes(stepper, dt, config.time)
    ScalarH5History(len, HDF5.create_dataset(h5_file, name, Float64, len))
end

#
# HDF5 vector field (4dim) history 
#
struct VectorFieldH5History <: AbstractVectorFieldHistory
    len::Int
    inner::HDF5.Dataset
end
function Base.setindex!(hist::VectorFieldH5History, value::ARR, idx::Int) where ARR <: AbstractArray{Float64, 4}
    if idx <= hist.len
        # x, y, z, component, vector write index
        hist.inner[:, :, :, :, idx] = value
    end
end
function vector_field_h5_history(stepper::STEP, h5_file, name::String, config::Configuration.Config{CONFIG})::VectorFieldH5History where STEP <: AbstractIoStepControl where CONFIG <: AbstractConfig
    dt = Configuration.calculate_dt(config)
    len = Io.num_writes(stepper, dt, config.time)
    N = config.N
    VectorFieldH5History(len, HDF5.create_dataset(h5_file, name, Float64, (N, N, N, 3, len)))
end

#################
################# Calculators for exported quantities
#################

#
# Energy
#
struct EnergyExport{ARR <: AbstractArray{Float64, 4}, HIST <: AbstractScalarHistory} <: AbstractIoExport
    N::Int
	stepper::Io.DtWrite
	history::HIST
	U::ARR
end

function Io.export_data(exporter::EnergyExport{ARR}, time::Float64) where ARR <: AbstractArray{Float64, 4}
	energy = sum(exporter.U .* exporter.U) / 2.
    energy *= (2 * pi/exporter.N)^3

    step = Io.step_number(exporter.stepper)
    exporter.history[step] = energy
end
	
function Io.get_stepper(exporter::EnergyExport{ARR, HIST}) where ARR <: AbstractArray{Float64, 4} where HIST <: AbstractScalarHistory
	exporter.stepper
end

#
# Helicity
#
struct HelicityExport{FARR <: AbstractArray{ComplexF64, 4}, ARR <: AbstractArray{Float64, 4}, P <: AbstractParallel, WAVE <: AbstractWavenumbers, HIST <: AbstractScalarHistory} <: AbstractIoExport 
    N::Int
    parallel::P
    K::WAVE
    plan::Plan
	stepper::Io.DtWrite
	history::HIST
	U_hat::FARR
	U::ARR
	omega::ARR
end

function Io.export_data(exporter::HelicityExport{FARR, ARR, P, K, HIST}, time::Float64
) where ARR <: AbstractArray{Float64, 4} where FARR <: AbstractArray{ComplexF64, 4} where P <: AbstractParallel where K <: AbstractWavenumbers where HIST<: AbstractScalarHistory

    solver.curl!(exporter.parallel, exporter.K, exporter.plan, exporter.U_hat; out = exporter.omega)
	helicity = sum(exporter.U .* exporter.omega) / 2.
    helicity *= (2 * pi/exporter.N)^3

    step = Io.step_number(exporter.stepper)
    exporter.history[step] = helicity
end
	
function Io.get_stepper(exporter::HelicityExport)
	exporter.stepper
end

#
# Time
#
struct TimeExport{HIST <: AbstractScalarHistory} <: AbstractIoExport 
	stepper::Io.DtWrite
	history::HIST
end

function Io.export_data(exporter::TimeExport{HIST}, time::Float64
) where HIST <: AbstractScalarHistory
    step = Io.step_number(exporter.stepper)
    exporter.history[step] = time
end
	
function Io.get_stepper(exporter::TimeExport)
	exporter.stepper
end

#
# Vector Fields
#
struct VectorFieldExport{HIST <: AbstractVectorFieldHistory, ARR <: AbstractArray{Float64, 4}} <: AbstractIoExport 
	stepper::Io.DtWrite
	history::HIST
    vector_field::ARR
end

function Io.export_data(exporter::VectorFieldExport{HIST, ARR}, time::Float64
) where HIST <: AbstractVectorFieldHistory where ARR <: AbstractArray{Float64, 4}
    step = Io.step_number(exporter.stepper)
    exporter.history[step] = exporter.vector_field
end
	
function Io.get_stepper(exporter::VectorFieldExport{HIST, ARR}
) where HIST <: AbstractVectorFieldHistory where ARR <: AbstractArray{Float64, 4}
	exporter.stepper
end

#
end
