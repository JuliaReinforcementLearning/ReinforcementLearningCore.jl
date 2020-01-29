export Agent

"""
    Agent(;kwargs...)

One of the most commonly used [`AbstractAgent`](@ref).

Generally speaking, it does nothing but

1. Pass observation to the policy to generate an action
1. Update the trajectory using the `observation => action` pair
1. Update the policy with the newly updated trajectory

# Keywords & Fields

- `policy`::[`AbstractPolicy`](@ref): the policy to use
- `trajectory`::[`AbstractTrajectory`](@ref): used to store transitions between agent and environment
- `role=DEFAULT_PLAYER`: used to distinguish different agents
"""
Base.@kwdef mutable struct Agent{P<:AbstractPolicy, B<:AbstractTrajectory, R} <: AbstractAgent
    policy::P
    trajectory::B
    role::R = DEFAULT_PLAYER
end

function (agent::Agent)(stage::AbstractStage, obs)
    update!(stage, agent.trajectory, obs)
end

function (agent::Agent)(stage::PreActStage, obs)
    action = agent.policy(obs)
    update!(stage, agent.trajectory, ObsAndAction(obs, action))
    update!(agent.policy, agent.trajectory)
    action
end

# update trajectory

function RLBase.update!(::PreActStage, trajectory::AbstractTrajectory, obs)
    push!(trajectory; state=get_state(obs), action=get_action(obs))
end

function RLBase.update!(::PostActStage, trajectory::AbstractTrajectory, obs)
    push!(trajectory; reward=get_reward(obs), terminal=get_terminal(obs))
end

RLBase.update!(::AbstractStage, trajectory::AbstractTrajectory, obs) = nothing