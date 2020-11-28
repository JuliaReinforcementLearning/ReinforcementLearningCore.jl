export Agent,
    role

import Functors:functor
using Setfield: @set

"""
    Agent(;kwargs...)

A wrapper of an `AbstractPolicy`. Generally speaking, it does nothing but to
update the trajectory and policy appropriately in different stages and modes.

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`AbstractTrajectory`](@ref): used to store transitions between an agent and an environment
- `role=RLBase.DEFAULT_PLAYER`: used to distinguish different agents
"""
Base.@kwdef struct Agent{P<:AbstractPolicy,T<:AbstractTrajectory,R,M} <: AbstractPolicy
    policy::P
    trajectory::T = DUMMY_TRAJECTORY
    role::R = RLBase.DEFAULT_PLAYER
    mode::M = TRAIN_MODE
end

functor(x::Agent) = (policy = x.policy,), y -> @set x.policy = y.policy

role(agent::Agent) = agent.role
mode(agent::Agent) = agent.mode
(agent::Agent)(env) = agent.policy(env)


(agent::Agent)(stage::AbstractStage, env::AbstractEnv) = agent(env, stage, mode(agent))

function (agent::Agent)(env::AbstractEnv, stage::AbstractStage, mode::AbstractMode)
    update!(agent.trajectory, agent.policy, env, stage, mode)
    update!(agent.policy, agent.trajectory, env, stage, mode)
end

## TrainMode

function (agent::Agent)(env::AbstractEnv, stage::PreActStage, mode::TrainMode)
    action = update!(agent.trajectory, agent.policy, env, stage, mode)
    update!(agent.policy, agent.trajectory, env, stage, mode)
    action
end

## EvalMode

function (agent::Agent)(env::AbstractEnv, stage::PreActStage, mode::EvalMode)
    update!(agent.trajectory, agent.policy, env, stage, mode)
end

## TestMode

(agent::Agent)(::AbstractEnv, ::AbstractStage, ::TestMode) = nothing
(agent::Agent)(env::AbstractEnv, ::PreActStage, ::TestMode) = agent.policy(env)

## update trajectory

function RLBase.update!(trajectory::AbstractTrajectory, ::AbstractPolicy, ::AbstractEnv, ::PreEpisodeStage, ::AbstractMode)
    if length(trajectory) > 0
        pop!(trajectory[:state])
        pop!(trajectory[:action])
        haskey(trajectory, :legal_actions_mask) && pop!(trajectory[:legal_actions_mask])
    end
end

function RLBase.update!(trajectory::AbstractTrajectory, policy::AbstractPolicy, env::AbstractEnv, ::PostEpisodeStage, ::AbstractMode)
    action = policy(env)
    push!(trajectory[:state], get_state(env))
    push!(trajectory[:action], action)
    haskey(trajectory, :legal_actions_mask) && push!(trajectory[:legal_actions_mask], get_legal_actions_mask(env))
end

function RLBase.update!(trajectory::CircularArraySARTTrajectory, policy::AbstractPolicy, env::AbstractEnv, ::PreActStage, ::AbstractMode)
    action = policy(env)
    push!(trajectory[:state], get_state(env))
    push!(trajectory[:action], action)
    haskey(trajectory, :legal_actions_mask) && push!(trajectory[:legal_actions_mask], get_legal_actions_mask(env))
    action
end

function RLBase.update!(trajectory::AbstractTrajectory, ::AbstractPolicy, env::AbstractEnv, ::PostActStage, ::AbstractMode)
    push!(trajectory[:reward], get_reward(env))
    push!(trajectory[:terminal], get_terminal(env))
end

## update policy

RLBase.update!(::AbstractPolicy, ::AbstractTrajectory, ::AbstractEnv, ::AbstractStage, ::AbstractMode) = nothing

RLBase.update!(policy::AbstractPolicy, trajectory::AbstractTrajectory, ::AbstractEnv, ::PreActStage, ::AbstractMode) = update!(policy, trajectory)
