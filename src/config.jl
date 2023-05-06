module config

export Config, create_config

struct Config
    ν::Float64
    re::Float64
    N::Int
end

function create_config(N::Int, re::Float64)::Config
    ν = 1 / re

    return Config(ν, re, N)
end

end
