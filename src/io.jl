module Io

export never_write, dt_write, step_number, should_write, increase_step

using ..markers: AbstractIoStepControl, AbstractIoExport
using Printf

struct NeverWrite <: AbstractIoStepControl
end

mutable struct DtWrite <: AbstractIoStepControl
    dt::Float64
    step_number::Int
    force_first_write::Bool
    last_floor::Float64
end

never_write()::NeverWrite = NeverWrite()

dt_write(dt::Float64)::DtWrite = DtWrite(dt, 1, true, 0.0)

function num_writes(stepper::DtWrite, dt::Float64, runtime::Float64)::Int
    runtime = runtime -= 1e-8

    if stepper.dt < dt
        # we will be writing the first data point no matter what
        stepper.force_first_write = false

        return Int(ceil(runtime / dt))
    else
        # tentatively the total amout of writes to input
        writes = Int(floor(runtime / stepper.dt))

        # if we are forcing a write for the first timestep, then
        # we have one additional write
        if stepper.force_first_write
            writes += 1
        end

        return writes
    end
end

function num_writes(stepper::NeverWrite, dt::Float64, runtime::Float64)::Int
    0
end

# get the current step that we are writing for
step_number(stepper::DtWrite) = stepper.step_number

# get the current step that we are writing for
step_number(stepper::NeverWrite) = error("step_number should never be called since this data should not be exported")

# check if we should write for this time step
function should_write(stepper::NeverWrite)::Bool
    false
end

# check if we should write for this time step
function should_write(stepper::DtWrite, time::Float64)::Bool
    # if it is the first timestep, and we wish to force the data to be written to a file...
    if stepper.force_first_write
        return true
    end

    new_floor = floor(time / stepper.dt)
    should_step = new_floor > stepper.last_floor

    stepper.last_floor = new_floor

    return should_step
end

# bump the step number for the next write
function increase_step!(stepper::DtWrite)
    stepper.step_number += 1

    # ensure this is false 
    stepper.force_first_write = false
end

# bump the step number for the next write (errors)
increase_step!(stepper::NeverWrite) = error("increase_step should never be called since this data should not be exported")

#
# Export functions
#

function get_stepper(exporter::EXPORT) where EXPORT <: AbstractIoExport
    error(@sprintf "get_stepper not implemented for exporter `%s` Ensure that you satisfy this interface." typeof(exporter))
end

function export_data(exporter::EXPORT, time::Float64) where EXPORT <: AbstractIoExport
    error(@sprintf "export_data not implemented for stepper `%s` and exporter `%s`. Ensure that you satisfy this interface." typeof(stepper), typeof(exporter))
end

#
end
