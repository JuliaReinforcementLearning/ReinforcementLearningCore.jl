export AbstractLearner

"""
    (learner::AbstractLearner)(obs)

A learner is usually used to estimate state values, state-action values or distributional values based on experiences.
"""
abstract type AbstractLearner end

function (learner::AbstractLearner)(obs) end

"""
    update!(learner::AbstractLearner, experience)

Typical `experience` is [`AbstractTrajectory`](@ref).
"""
function RLBase.update!(learner::AbstractLearner, experience) end

"""
    get_priority(p::AbstractLearner, experience)
"""
function RLBase.get_priority(p::AbstractLearner, experience) end