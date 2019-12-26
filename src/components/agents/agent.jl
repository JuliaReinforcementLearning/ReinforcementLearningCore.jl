export Agent

"""
    Agent(;kwargs...)

One of the most commonly used [`AbstractAgent`](@ref).

Generally speaking, it does nothing but

1. Pass observation to the policy to generate an action
1. Update the buffer using the `observation => action` pair
1. Update the policy with the newly updated buffer

# Keywords & Fields

- `π`::[`AbstractPolicy`](@ref): the policy to use
- `buffer`::[`AbstractTrajectory`](@ref): used to store transitions between agent and environment
- `role=:DEFAULT`: used to distinguish different agents
"""
Base.@kwdef mutable struct Agent{P<:AbstractPolicy, B<:AbstractTrajectory, R} <: AbstractAgent
    π::P
    buffer::B
    role::R = DEFAULT_PLAYER
end

function (agent::Agent)(::PreEpisodeStage, obs)
    action = agent.π(obs)
    push!(agent.buffer; state=get_state(obs), action=action)
    update!(agent.π, agent.buffer)
    action
end

function (agent::Agent)(::PreActStage, obs)
    action = agent.π(obs)
    push!(agent.buffer; reward=get_reward(obs), terminal=is_terminal(obs), next_state=get_state(obs), next_action=action)
    update!(agent.π, agent.buffer)
    action
end

(agent::Agent)(::PostActStage, obs) = nothing

function (agent::Agent)(::PostEpisodeStage, obs)
    push!(agent.buffer; reward=get_reward(obs), terminal=is_terminal(obs), next_state=get_state(obs), next_action=agent.π(obs))
    update!(agent.π, agent.buffer)
    nothing
end