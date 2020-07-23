export ChancePlayerPolicy

using Random

struct ChancePlayerPolicy <: AbstractPolicy
    rng::AbstractRNG
end

ChancePlayerPolicy(;rng=Random.GLOBAL_RNG) = ChancePlayerPolicy(rng)

function (p::ChancePlayerPolicy)(env)
    v = rand(p.rng)
    s = 0.
    for (action, prob) in get_chance_outcome(env)
        s += prob
        s >= v && return action
    end
end
