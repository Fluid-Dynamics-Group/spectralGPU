module spectralGPU

include("./markers.jl")
include("./mesh.jl")
include("./fft.jl")
include("./initial_condition.jl")
include("./state.jl")
include("./config.jl")

include("./solver.jl")

export fft
export mesh
export markers
export state
export config
export initial_condition

#
end
