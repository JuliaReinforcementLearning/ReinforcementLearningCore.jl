export AbstractLearner, extract_experience

using Flux

"""
    (learner::AbstractLearner)(obs)

A learner is usually used to estimate state values, state-action values or distributional values based on experiences.
"""
abstract type AbstractLearner end

function (learner::AbstractLearner)(obs) end

"""
    get_priority(p::AbstractLearner, experience)
"""
function RLBase.get_priority(p::AbstractLearner, experience) end

Flux.testmode!(learner::AbstractLearner, mode=true) = Flux.testmode!(learner.approximator, mode)