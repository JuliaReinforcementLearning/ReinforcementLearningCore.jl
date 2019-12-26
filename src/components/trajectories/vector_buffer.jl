export VectorBuffer, VectorSARTSABuffer

const VectorBuffer = Trajectory{names, types, NamedTuple{names, trace_types}} where {names, types, trace_types<:Tuple{Vararg{<:Vector}}}

function VectorBuffer(;kwargs...)
    names = keys(kwargs.data)
    types = values(kwargs.data)
    buffers = merge(NamedTuple(), (name, Vector{type}()) for (name, type) in zip(names, types))
    Trajectory{names,Tuple{types...},typeof(buffers)}(buffers)
end

RLBase.isfull(b::VectorBuffer) = false

const VectorSARTSABuffer = Trajectory{SARTSA, types, NamedTuple{SARTSA, trace_types}} where {types, trace_types<:Tuple{Vararg{<:Vector}}}

function VectorSARTSABuffer(;state_type=Int, action_type=Int, reward_type=Float64, terminal_type=Bool, next_state_type=state_type,next_action_type=action_type)
    VectorBuffer(;
        state=state_type,
        action=action_type,
        reward=reward_type,
        terminal=terminal_type,
        next_state=next_state_type,
        next_action=next_action_type)
end