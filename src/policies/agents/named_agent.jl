export NamedAgent

import Functors: functor
using Setfield: @set

"""
    NamedAgent(name=>policy)

A policy wrapper to provide a name. Mostly used in multi-agent environments.
"""
struct NamedAgent{P,N} <: AbstractPolicy
    name::N
    policy::P
end

NamedAgent((name,policy)) = NamedAgent(policy, name)

functor(x::NamedAgent) = (policy = x.policy,), y -> @set x.policy = y.policy

Base.nameof(agent::NamedAgent) = agent.name

function check(agent::NamedAgent, env::AbstractEnv)
    check(agent.policy, env)
end
