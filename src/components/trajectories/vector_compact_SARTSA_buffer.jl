export VectorCompactSARTSABuffer

const VectorCompactSARTSABuffer = Trajectory{SARTSA, types, NamedTuple{RTSA, trace_types}, Val{:unlimited}} where {types, trace_types<:Tuple{Vararg{<:Vector}}}

function VectorCompactSARTSABuffer(;state_type=Int, action_type=Int, reward_type=Float32, terminal_type=Bool)
    VectorCompactSARTSABuffer{
        Tuple{
            state_type,
            action_type,
            reward_type,
            terminal_type,
            state_type,
            action_type,
        },
        Tuple{
            Vector{reward_type},
            Vector{terminal_type},
            Vector{state_type},
            Vector{action_type},
        }
    }(
        (
            reward=Vector{reward_type}(),
            terminal = Vector{terminal_type}(),
            state=Vector{state_type}(),
            action=Vector{action_type}()
        ),
        Val(:unlimited)
    )
end

Base.length(b::VectorCompactSARTSABuffer) = max(0, length(getfield(getfield(b, :buffers), :state)) - 1)

RLBase.get_trace(b::VectorCompactSARTSABuffer, s::Symbol) = _get_trace(b, Val(s))
_get_trace(b::VectorCompactSARTSABuffer, ::Val{:state}) = select_last_dim(b[:state], 1:(length(b) == 0 ? length(b[:state]) : length(b[:state])-1))
_get_trace(b::VectorCompactSARTSABuffer, ::Val{:action}) = select_last_dim(b[:action], 1:(length(b) == 0 ? length(b[:state]) : length(b[:action])-1))
_get_trace(b::VectorCompactSARTSABuffer, ::Val{:reward}) = b[:reward]
_get_trace(b::VectorCompactSARTSABuffer, ::Val{:terminal}) = b[:terminal]
_get_trace(b::VectorCompactSARTSABuffer, ::Val{:next_state}) = select_last_dim(b[:state], 2:length(b[:state]))
_get_trace(b::VectorCompactSARTSABuffer, ::Val{:next_action}) = select_last_dim(b[:action], 2:length(b[:action]))

RLBase.isfull(b::VectorCompactSARTSABuffer) = false

function Base.empty!(b::VectorCompactSARTSABuffer)
    for s in RTSA
        empty!(b[s])
    end
    b
end

Base.isempty(b::VectorCompactSARTSABuffer) = all(isempty(b[s]) for s in RTSA)

function Base.push!(b::VectorCompactSARTSABuffer; state=nothing, action=nothing, reward=nothing, terminal=nothing, next_state=nothing, next_action=nothing)
    isnothing(state) || push!(b[:state], state)
    isnothing(action) || push!(b[:action], action)
    isnothing(reward) || push!(b[:reward], reward)
    isnothing(terminal) || push!(b[:terminal], terminal)
    isnothing(next_state) || push!(b[:state], next_state)
    isnothing(next_action) || push!(b[:action], next_action)
    b
end