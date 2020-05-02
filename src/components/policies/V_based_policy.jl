export VBasedPolicy

using MacroTools: @forward

"""
    VBasedPolicy(;learner, mapping, explorer=GreedyExplorer())

# Key words & Fields

- `learner`::[`AbstractLearner`](@ref), learn how to estimate state values.
- `mapping`, a customized function `(obs, learner) -> action_values`
- `explorer`::[`AbstractExplorer`](@ref), decide which action to take based on action values.
"""
Base.@kwdef struct VBasedPolicy{L<:AbstractLearner,M,E<:AbstractExplorer} <: AbstractPolicy
    learner::L
    mapping::M
    explorer::E = GreedyExplorer()
end

(p::VBasedPolicy)(obs, ::MinimalActionSet) = p.mapping(obs, p.learner) |> p.explorer

function (p::VBasedPolicy)(obs, ::FullActionSet)
    action_values = p.mapping(obs, p.learner)
    p.explorer(action_values, get_legal_actions_mask(obs))
end

RLBase.get_prob(p::VBasedPolicy, obs, ::MinimalActionSet) = get_prob(p.explorer, p.mapping(obs, p.learner))

function RLBase.get_prob(p::VBasedPolicy, obs, ::FullActionSet)
    action_values = p.mapping(obs, p.learner)
    get_prob(p.explorer, action_values, get_legal_actions_mask(obs))
end

@forward VBasedPolicy.learner RLBase.get_priority, RLBase.update!