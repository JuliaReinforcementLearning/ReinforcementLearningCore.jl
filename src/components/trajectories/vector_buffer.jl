export VectorBuffer, VectorSARTSABuffer

const VectorBuffer = Trajectory{names, types, NamedTuple{names, trace_types}, Val{:unlimited}} where {names, types, trace_types<:Tuple{Vararg{<:Vector}}}

function VectorBuffer(;kwargs...)
    names = keys(kwargs.data)
    types = values(kwargs.data)
    buffers = merge(NamedTuple(), (name, Vector{type}()) for (name, type) in zip(names, types))
    VectorBuffer{names,Tuple{types...},typeof(values(buffers))}(buffers, Val(:unlimited))
end

RLBase.isfull(b::VectorBuffer) = false