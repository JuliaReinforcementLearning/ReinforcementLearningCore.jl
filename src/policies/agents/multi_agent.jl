export MultiAgent, NO_OP, NoOp

"Represent no-operation if it's not the agent's turn."
struct NoOp end

const NO_OP = NoOp()

struct MultiAgent <: AbstractPolicy
    agents::Dict{Any,Any}
end

Base.getindex(A::MultiAgent, x) = getindex(A.agents, x)

"""
    MultiAgent(player => policy...)

This is the simplest form of multiagent system. At each step they observe the
environment from their own perspective and get updated independently. For
environments of `SEQUENTIAL` style, agents which are not the current player will
observe a dummy action of [`NO_OP`](@ref) in the `PreActStage`.
"""
MultiAgent(policies...) = MultiAgent(Dict(nameof(p) => p for p in policies))

(A::MultiAgent)(env::AbstractEnv) = A(env, DynamicStyle(env))
(A::MultiAgent)(env::AbstractEnv, ::Sequential) = A[current_player(env)](env)
(A::MultiAgent)(env::AbstractEnv, ::Simultaneous) = [agent(env) for agent in values(A.agents)]

function (A::MultiAgent)(stage::AbstractStage, env::AbstractEnv)
    for agent in values(A.agents)
        agent(stage, env)
    end
end

function (A::MultiAgent)(stage::PreActStage, env::AbstractEnv, action)
    A(stage, env, DynamicStyle(env), action)
end

function (A::MultiAgent)(stage::PreActStage, env::AbstractEnv, ::Sequential, action)
    p = current_player(env)
    for (player, agent) in A.agents
        if p == player
            agent(stage, env, action)
        else
            agent(stage, env, NO_OP)
        end
    end
end

function (A::MultiAgent)(stage::PreActStage, env::AbstractEnv, ::Simultaneous, actions)
    for (agent, action) in zip(values(A.agents), actions)
        agent(stage, env, action)
    end
end