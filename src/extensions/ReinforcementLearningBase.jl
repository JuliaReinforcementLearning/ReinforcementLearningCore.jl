using CUDA
using Distributions: pdf
using Random
using Flux
using BSON

Random.rand(s::MultiContinuousSpace{<:CuArray}) = rand(CUDA.CURAND.generator(), s)

# avoid fallback silently
Flux.testmode!(p::AbstractPolicy, mode = true) =
    @error "someone forgets to implement this method!!!"

function save(f::String, p::AbstractPolicy)
    policy = cpu(p)
    BSON.@save f policy
end

function load(f::String, ::Type{<:AbstractPolicy})
    BSON.@load f policy
    policy
end
