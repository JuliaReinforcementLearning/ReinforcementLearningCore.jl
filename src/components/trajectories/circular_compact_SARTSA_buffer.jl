export CircularCompactSARTSABuffer

const CircularCompactSARTSABuffer = Trajectory{SARTSA, T1, NamedTuple{RTSA, T2}, Val{:circular}} where {T1, T2<:Tuple{Vararg{<:CircularArrayBuffer}}}

function CircularCompactSARTSABuffer(;
    capacity,
    state_type = Int,
    state_size = (),
    action_type = Int,
    action_size = (),
    reward_type = Float64,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
)
    capacity > 0 || throw(ArgumentError("capacity must > 0"))
    CircularCompactSARTSABuffer{ 
        Tuple{
            state_type,
            action_type,
            reward_type,
            terminal_type,
            state_type,
            action_type
        },
        Tuple{
            CircularArrayBuffer{reward_type, length(reward_size)+1},
            CircularArrayBuffer{terminal_type, length(terminal_size)+1},
            CircularArrayBuffer{state_type, length(state_size)+1},
            CircularArrayBuffer{action_type, length(action_size)+1}
        }
    }(
        (
            reward = CircularArrayBuffer{reward_type}(reward_size..., capacity),
            terminal = CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
            state = CircularArrayBuffer{state_type}(state_size..., capacity+1),
            action = CircularArrayBuffer{action_type}(action_size..., capacity+1)
        ),
        Val(:circular)
    )
end

Base.length(b::CircularCompactSARTSABuffer) = length(b[:terminal])
Base.isempty(b::CircularCompactSARTSABuffer) = all(isempty(b[s]) for s in RTSA)
RLBase.isfull(b::CircularCompactSARTSABuffer) = all(isfull(b[s]) for s in RTSA)

"Exactly the same with [`EpisodeCompactSARTSABuffer`](@ref)"
RLBase.get_trace(b::CircularCompactSARTSABuffer, s::Symbol) = _get_trace(b, Val(s))
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:state}) = select_last_dim(b[:state], 1:(length(b) == 0 ? length(b[:state]) : length(b[:state])-1))
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:action}) = select_last_dim(b[:action], 1:(length(b) == 0 ? length(b[:state]) : length(b[:action])-1))
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:reward}) = b[:reward]
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:terminal}) = b[:terminal]
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:next_state}) = select_last_dim(b[:state], 2:length(b[:state]))
_get_trace(b::CircularCompactSARTSABuffer, ::Val{:next_action}) = select_last_dim(b[:action], 2:length(b[:action]))

function Base.getindex(b::CircularCompactSARTSABuffer, i::Int)
    (
        state = select_last_dim(b[:state], i),
        action = select_last_dim(b[:action], i),
        reward = select_last_dim(b[:reward], i),
        terminal = select_last_dim(b[:terminal], i),
        next_state = select_last_dim(b[:state], i+1),
        next_action = select_last_dim(b[:action], i+1)
    )
end

function Base.empty!(b::CircularCompactSARTSABuffer)
    for s in RTSA
        empty!(b[s])
    end
    b
end

Base.push!(b::CircularCompactSARTSABuffer; state=nothing, action=nothing, reward=nothing, terminal=nothing) = push!(b, state, action, reward, terminal)

function Base.push!(b::CircularCompactSARTSABuffer, s, a, ::Nothing, ::Nothing)
    push!(b[:state], s)
    push!(b[:action], a)
    b
end

function Base.push!(b::CircularCompactSARTSABuffer, ::Nothing, ::Nothing, r, t)
    push!(b[:reward], r)
    push!(b[:terminal], t)
    b
end

function Base.pop!(b::CircularCompactSARTSABuffer, ::Val{:state}, ::Val{:action})
    pop!(b[:state])
    pop!(b[:action])
end