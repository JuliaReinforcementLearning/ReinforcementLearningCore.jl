export AbstractLearner, extract_experience

using Flux

"""
    (learner::AbstractLearner)(env)

A learner is usually used to estimate state values, state-action values or distributional values based on experiences.
"""
abstract type AbstractLearner end

Base.summary(io::IO, t::T) where T<:AbstractLearner = print(io, T.name)

function (learner::AbstractLearner)(env) end

"""
    get_priority(p::AbstractLearner, experience)
"""
function RLBase.get_priority(p::AbstractLearner, experience) end

Flux.testmode!(learner::AbstractLearner, mode = true) =
    Flux.testmode!(learner.approximator, mode)
