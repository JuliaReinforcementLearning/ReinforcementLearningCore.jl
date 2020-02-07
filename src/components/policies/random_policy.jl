export RandomPolicy

using Random

Base.@kwdef struct RandomPolicy{S<:AbstractSpace,R<:AbstractRNG} <: AbstractPolicy
    action_space::S
    rng::R = MersenneTwister()
end

Base.show(io::IO, p::RandomPolicy) = print(io, "RandomPolicy($(p.action_space))")

Random.seed!(p::RandomPolicy, seed) = Random.seed!(p.rng, seed)

RandomPolicy(env::AbstractEnv; seed = nothing) =
    RandomPolicy(; action_space = get_action_space(env), rng = MersenneTwister(seed))

(p::RandomPolicy)(obs, ::FullActionSet) = rand(p.rng, get_legal_actions(obs))
(p::RandomPolicy)(obs, ::MinimalActionSet) = rand(p.rng, p.action_space)

RLBase.update!(p::RandomPolicy, experience) = nothing
