export OffPolicy

using MacroTools: @forward

"""
    OffPolicy(π_target::P, π_behavior::B) -> OffPolicy{P,B}
"""
Base.@kwdef struct OffPolicy{P,B} <: AbstractPolicy
    π_target::P
    π_behavior::B
end

(π::OffPolicy)(obs) = π.π_behavior(obs)

struct WeightedExperience{E, W}
    experience::E
    weight::W
end

function RLBase.update!(π::OffPolicy, experience)
    weight = get_prob(π.π_target, experience) ./ get_prob(π.π_behavior, experience)
    update!(π.π_target, WeightedExperience(experience, weight))
end

@forward OffPolicy.π_behavior RLBase.get_priority, RLBase.get_prob