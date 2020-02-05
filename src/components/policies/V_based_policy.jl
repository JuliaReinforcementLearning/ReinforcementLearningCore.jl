export VBasedPolicy

"""
# Fields

- `value_learner`::[`AbstractLearner`](@ref), learn how to estimate state values.
- `mapping`, an arbitrary function, used to transform the state values into action values.
- `explorer`::[`AbstractExplorer`](@ref), decide which action to take based on action values.
"""
struct VBasedPolicy{L<:AbstractLearner, M, E<:AbstractExplorer} <: AbstractPolicy
    value_learner::L
    mapping::M
    explorer::E
end

(p::VBasedPolicy)(obs) = obs |> p.value_learner |> p.mapping |> p.explorer

RLBase.update!(p::VBasedPolicy, experience) = update!(p.value_learner, experience)

RLBase.extract_experience(trajectory<:AbstractTrajectory, p::VBasedPolicy) = extract_experience(trajectory, p.value_learner)