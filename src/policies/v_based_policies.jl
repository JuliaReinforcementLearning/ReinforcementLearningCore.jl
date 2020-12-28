export VBasedPolicy

function default_value_action_mapping(env, value_learner)
    A = legal_action_space(env)
    a, v = A[1], -Inf
    for a′ in A
        v′ = value_learner(child(env, a))
        if v′ > v
            a = a′
            v = v′
        end
    end
    a
end

"""
    VBasedPolicy(;learner, mapping=default_value_action_mapping)

The `learner` must be a value learner. The `mapping` is a function which returns
an action given `env` and the `learner`. By default we iterate through all the
valid actions and select the best one which lead to the maximum state value.
"""
Base.@kwdef struct VBasedPolicy{L, M} <: AbstractPolicy
    learner::L
    mapping::M = default_value_action_mapping
end

(p::VBasedPolicy)(env::AbstractEnv) = p.mapping(env, p.learner)

RLBase.update!(p::VBasedPolicy, args...) = update!(p.learner, args...)
