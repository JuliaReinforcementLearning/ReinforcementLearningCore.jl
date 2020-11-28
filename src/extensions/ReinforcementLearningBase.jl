using CUDA
using Distributions: pdf
using Random
using Flux
using BSON
using AbstractTrees

RLBase.update!(p::RandomPolicy, x) = nothing

Random.rand(s::MultiContinuousSpace{<:CuArray}) = rand(CUDA.CURAND.generator(), s)

Base.show(io::IO, p::AbstractPolicy) =
    AbstractTrees.print_tree(io, StructTree(p), get(io, :max_depth, 10))

is_expand(::AbstractEnv) = false

AbstractTrees.printnode(io::IO, t::StructTree{<:AbstractEnv}) = print(
    io,
    "$(RLBase.get_name(t.x)): $(join([f(t.x) for f in RLBase.get_env_traits()], ","))",
)
