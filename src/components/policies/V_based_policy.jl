export VBasedPolicy

"""
# Fields

- `value_learner`::[`AbstractLearner`](@ref), learn how to estimate state values.
- `mapping`, a customized function `(obs, value_learner) -> action_values`
- `explorer`::[`AbstractExplorer`](@ref), decide which action to take based on action values.
"""
Base.@kwdef struct VBasedPolicy{L<:AbstractLearner, M, E<:AbstractExplorer} <: AbstractPolicy
    value_learner::L
    mapping::M
    explorer::E = GreedyExplorer()
end

(p::VBasedPolicy)(obs, ::MinimalActionSet) = p.mapping(obs, p.value_learner) |> p.explorer

function (p::VBasedPolicy)(obs, ::FullActionSet)
    action_values = p.mapping(obs, p.value_learner) 
    p.explorer(action_values, get_legal_actions_mask(obs))
end

function RLBase.get_prob(p::VBasedPolicy, obs, ::MinimalActionSet)
    get_prob(p.explorer, p.mapping(obs, p.value_learner))
end

function RLBase.get_prob(p::VBasedPolicy, obs, ::FullActionSet)
    action_values = p.mapping(obs, p.value_learner) 
    get_prob(p.explorer, action_values, get_legal_actions_mask(obs))
end

function RLBase.update!(p::VBasedPolicy, t::AbstractTrajectory)
    experience = extract_experience(t, p)
    update!(p.value_learner, experience)
end

RLBase.extract_experience(trajectory::AbstractTrajectory, p::VBasedPolicy) = extract_experience(trajectory, p.value_learner)