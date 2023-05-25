module Exporters

export EnergyExport

using ..markers: AbstractIoExport, AbstractParallel, AbstractWavenumbers
using ..fft: Plan
import ..solver
import ..Io

struct EnergyExport{ARR <: AbstractArray{Float64, 4}} <: AbstractIoExport
    N::Int
	stepper::Io.DtWrite
	history::Vector{Float64}
	U::ARR
end

function Io.export_data(exporter::EnergyExport{ARR}, time::Float64) where ARR <: AbstractArray{Float64, 4}
	energy = sum(exporter.U .* exporter.U) / 2.
    energy *= (2 * pi/exporter.N)^3
	push!(exporter.history, energy)
end
	
function Io.get_stepper(exporter::EnergyExport{ARR}) where ARR <: AbstractArray{Float64, 4}
	exporter.stepper
end

struct HelicityExport{FARR <: AbstractArray{ComplexF64, 4}, ARR <: AbstractArray{Float64, 4}, P <: AbstractParallel, WAVE <: AbstractWavenumbers} <: AbstractIoExport
    N::Int
    parallel::P
    K::WAVE
    plan::Plan
	stepper::Io.DtWrite
	history::Vector{Float64}
	U_hat::FARR
	U::ARR
	omega::ARR
end

function Io.export_data(exporter::HelicityExport{FARR, ARR, P, K}, time::Float64
) where ARR <: AbstractArray{Float64, 4} where FARR <: AbstractArray{ComplexF64, 4} where P <: AbstractParallel where K <: AbstractWavenumbers

    solver.curl!(exporter.parallel, exporter.K, exporter.plan, exporter.U_hat; out = exporter.omega)
	helicity = sum(exporter.U .* exporter.omega) / 2.
    helicity *= (2 * pi/exporter.N)^3
	push!(exporter.history, helicity)
end
	
function Io.get_stepper(exporter::HelicityExport)
	exporter.stepper
end


end
