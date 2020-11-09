export TrajectoryClient

using MacroTools: @forward

#####
# Client part
#####

mutable struct TrajectoryClient{T<:AbstractTrajectory,A,S} <: AbstractTrajectory
    trajectory::T
    adder::A
    mailbox::S
    sync_freq::Int
    n::Int
end

@forward TrajectoryClient.trajectory Base.keys,
Base.haskey,
Base.getindex,
Base.pop!,
Base.empty!,
isfull

function Base.push!(t::TrajectoryClient, args...;kwargs...)
    push!(t.trajectory, args...;kwargs...)
    t.n += 1

    if t.n % t.sync_freq == 0
        put!(t.mailbox, deepcopy(t.trajectoryj))
    end
end


#####
# TrajectorySampler
#####

abstract type AbstractAdder end

Base.@kwdef struct NStepAdder <: AbstractAdder
    n::Int = 1
end

#####
# Server part
#####

function Base.push!(t::VectSARTSATrajectory, 𝕥::CircularCompactSARTSATrajectory, adder::NStepAdder)
    N = length(𝕥[:terminal])
    n = adder.n
    for i in 1:(N-n+1)
        push!(
            t;
            state=select_last_dim(𝕥[:state], i:i+n-1),
            action=select_last_dim(𝕥[:action], i:i+n-1),
            reward=select_last_dim(𝕥[:reward], i:i+n-1),
            terminal=select_last_dim(𝕥[:terminal], i:i+n-1),
            next_state=select_last_dim(𝕥[:next_state], i:i+n-1),
            next_action=select_last_dim(𝕥[:next_action], i:i+n-1),
            )
    end
end