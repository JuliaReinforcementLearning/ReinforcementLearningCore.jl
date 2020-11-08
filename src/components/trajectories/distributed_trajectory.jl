export TrajectoryClient

using MacroTools: @forward

#####
# Client part
#####

struct TrajectoryClient{T<:AbstractTrajectory, S} <: AbstractTrajectory
    trajectory::T
    bulk_size::Int
    mailbox::S
end

@forward TrajectoryClient.trajectory Base.keys,
Base.haskey,
Base.getindex,
Base.pop!,
Base.empty!,
isfull

function Base.push!(t::TrajectoryClient, args...;kwargs...)
    push!(t.trajectory, args...;kwargs...)
    _sync(t)
end

# Given that CircularCompactSARTSATrajectory is the most common one
# We'll focus on the implementations around it for now

function _sync(t::TrajectoryClient{CircularCompactSARTSATrajectory})
    if nframes(t.trajectory[:full_state]) >= t.bulk_size
        # TODO: here we simply create an copy to avoid sharing the same data accross different tasks
        # But for remote channels, this is redundant because it will always copy the data.
        d = deepcopy(t.trajectory)
        put!(t.mailbox, d)
    end
end

#####
# Server part
#####

function Base.push!(t::VectSARTSATrajectory, 𝕥::CircularCompactSARTSATrajectory)
    for i in 1:length(𝕥[:terminal])
        push!(
            t;
            state=select_last_dim(𝕥[:state], i),
            action=select_last_dim(𝕥[:action], i),
            reward=select_last_dim(𝕥[:reward], i),
            terminal=select_last_dim(𝕥[:terminal], i),
            next_state=select_last_dim(𝕥[:next_state], i),
            next_action=select_last_dim(𝕥[:next_action], i),
            )
    end
end