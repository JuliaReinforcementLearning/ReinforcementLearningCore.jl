using CUDA
using Distributions: pdf
using Random
using Flux
using BSON
using AbstractTrees


Random.rand(s::MultiContinuousSpace{<:CuArray}) = rand(CUDA.CURAND.generator(), s)

# avoid fallback silently
Flux.testmode!(p::AbstractPolicy, mode = true) =
    @error "someone forgets to implement this method!!!"

Base.show(io::IO, p::AbstractPolicy) = AbstractTrees.print_tree(io, StructTree(p),get(io, :max_depth, 10))

Base.summary(io::IO, t::T) where T<:AbstractPolicy = print(io, T.name)

function save(f::String, p::AbstractPolicy)
    policy = cpu(p)
    BSON.@save f policy
end

function load(f::String, ::Type{<:AbstractPolicy})
    BSON.@load f policy
    policy
end
