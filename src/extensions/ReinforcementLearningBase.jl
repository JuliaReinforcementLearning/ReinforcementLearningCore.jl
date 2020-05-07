using CuArrays
using Distributions: pdf
using Random
using Flux

RLBase.get_prob(p::AbstractPolicy, obs, ::RLBase.AbstractActionStyle, a) =
    pdf(get_prob(p, obs), a)

Random.rand(s::MultiContinuousSpace{<:CuArray}) = rand(CuArrays.CURAND.generator(), s)

# avoid fallback silently
Flux.testmode!(p::AbstractPolicy, mode = true) =
    @error "someone forgets to implement this method!!!"
