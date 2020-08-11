using MacroTools:@forward

#####
# Trace
#####

"""
    Trace(;[trace_name=trace_container]...)

Simply a wrapper of `NamedTuple`.
Define our own type here to avoid type piracy with `NamedTuple`
"""
struct Trace{T} <: AbstractTrace
    traces::T
end

Trace(;kwargs...) = Trace(kwargs.data)

@forward Trace.traces Base.keys, Base.haskey, Base.getindex

Base.push!(t::Trace, kv::Pair{Symbol}) = push!(t[first(kv)], last(kv))
Base.pop!(t::Trace, s::Symbol) = pop!(t[s])

#####
# SharedTrace
#####

struct SharedTraceMeta
    start_shift::Int
    end_shift::Int
end

"""
    SharedTrace(trace;[trace_name=start_shift:end_shift]...)

Create multiple traces sharing the same underlying container.
"""
struct SharedTrace{X,M} <: AbstractTrace
    x::X
    meta::M
end

function SharedTrace(x, s::Symbol)
    SharedTrace(
        x,
        (;
            s=>SharedTraceMeta(1, -1),
            Symbol(:next_, s)=>SharedTraceMeta(2, 0),
            Symbol(:full_, s) => SharedTraceMeta(1,0)
        )
    )
end

@forward SharedTrace.meta Base.keys, Base.haskey

function Base.getindex(t::SharedTrace, s::Symbol)
    m = t.meta[s]
    select_last_dim(t.x, m.start_shift:(nframes(t.x)+m.end_shift))
end

Base.push!(t::SharedTrace, kv::Pair{Symbol}) = push!(t.x, last(kv))
Base.empty!(t::SharedTrace) = empty!(t.x)
Base.pop!(t::SharedTrace, s::Symbol) = pop!(t.x)

function Base.pop!(t::SharedTrace)
    s = first(keys(t))
    (;s => pop!(t.x))
end

#####
# EpisodicTrace
#####

"""
Assuming that the `flag_trace` is in `traces` and it's an `AbstractVector{Bool}`, 
meaning whether an environment reaches terminal or not. The last element in
`flag_trace` will be used to determine whether the whole trace is full or not.
"""
struct EpisodicTrace{T, flag_trace} <: AbstractTrace
    traces::T
end

EpisodicTrace(traces::T, flag_trace=:terminal) where T = EpisodicTrace{T, flag_trace}(traces)

@forward EpisodicTrace.traces Base.keys, Base.haskey, Base.getindex, Base.push!, Base.pop!, Base.empty!

function isfull(t::EpisodicTrace{<:Any, F}) where F
    x = t.traces[F]
    (nframes(x) > 0) && select_last_frame(x)
end

#####
# CombinedTrace
#####

struct CombinedTrace{T1, T2} <: AbstractTrace
    t1::T1
    t2::T2
end

Base.haskey(t::CombinedTrace, s::Symbol) = haskey(t.t1, s) || haskey(t.t2, s)
Base.getindex(t::CombinedTrace, s::Symbol) = if haskey(t.t1, s)
    getindex(t.t1, s)
elseif haskey(t.t2, s)
    getindex(t.t2, s)
else
    throw(ArgumentError("unknown key: $s"))
end

Base.keys(t::CombinedTrace) = (keys(t.t1)..., keys(t.t2)...)

Base.push!(t::CombinedTrace, kv::Pair{Symbol}) = if haskey(t.t1, first(kv))
    push!(t.t1, kv)
elseif haskey(t.t2, first(kv))
    push!(t.t2, kv)
else
    throw(ArgumentError("unknown kv: $kv"))
end

Base.pop!(t::CombinedTrace, s::Symbol) = if haskey(t.t1, s)
    pop!(t.t1, s)
elseif haskey(t.t2, s)
    pop!(t.t2, s)
else
    throw(ArgumentError("unknown key: $s"))
end

Base.pop!(t::CombinedTrace) = merge(pop!(t.t1), pop!(t.t2))

function Base.empty!(t::CombinedTrace)
    empty!(t.t1)
    empty!(t.t2)
end

#####
# CircularCompactSATrace 
#####

const CircularCompactSATrace = CombinedTrace{
    <:SharedTrace{<:CircularArrayBuffer, <:NamedTuple{(:state, :next_state, :full_state)}},
    <:SharedTrace{<:CircularArrayBuffer, <:NamedTuple{(:action, :next_action, :full_action)}},
}

function CircularCompactSATrace(;
    capacity,
    state_type = Int,
    state_size = (),
    action_type = Int,
    action_size = (),
)
    CombinedTrace(
        SharedTrace(
            CircularArrayBuffer{state_type}(state_size..., capacity+1),
            :state),
        SharedTrace(
            CircularArrayBuffer{action_type}(action_size..., capacity+1),
            :action
        ),
    )
end

#####
# CircularCompactSALTrace 
#####

const CircularCompactSALTrace = CombinedTrace{
    <:SharedTrace{<:CircularArrayBuffer, <:NamedTuple{(:legal_actions_mask, :next_legal_actions_mask, :full_legal_actions_mask)}},
    <:CircularCompactSATrace
}

function CircularCompactSALTrace(;
    capacity,
    legal_actions_mask_size,
    legal_actions_mask_type=Bool,
    kw...
)
    CombinedTrace(
        SharedTrace(
            CircularArrayBuffer{legal_actions_mask_type}(legal_actions_mask_size..., capacity+1),
            :legal_actions_mask
        ),
        CircularCompactSATrace(;capacity=capacity, kw...)
    )
end
#####
# CircularCompactSARTSATrace
#####

const CircularCompactSARTSATrace = CombinedTrace{
    <:Trace{<:NamedTuple{(:reward, :terminal), <:Tuple{<:CircularArrayBuffer, <:CircularArrayBuffer}}},
    <:CircularCompactSATrace
}

function CircularCompactSARTSATrace(;
    capacity,
    reward_type = Float32,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
    kw...
)
    CombinedTrace(
        Trace(
            reward=CircularArrayBuffer{reward_type}(reward_size..., capacity),
            terminal=CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
        ),
        CircularCompactSATrace(;capacity=capacity, kw...),
    )
end

#####
# CircularCompactSALRTSALTrace
#####

const CircularCompactSALRTSALTrace = CombinedTrace{
    <:Trace{<:NamedTuple{(:reward, :terminal), <:Tuple{<:CircularArrayBuffer, <:CircularArrayBuffer}}},
    <:CircularCompactSALTrace
}

function CircularCompactSALRTSALTrace(;
    capacity,
    reward_type = Float32,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
    kw...
    )
    CombinedTrace(
        Trace(
            reward=CircularArrayBuffer{reward_type}(reward_size..., capacity),
            terminal=CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
        ),
        CircularCompactSALTrace(;capacity=capacity, kw...),
    )
end

#####
# CircularCompactPSARTSATrace
#####

const CircularCompactPSARTSATrace = CombinedTrace{
    <:Trace{<:NamedTuple{(:reward, :terminal,:priority), <:Tuple{<:CircularArrayBuffer, <:CircularArrayBuffer, <:SumTree}}},
    <:CircularCompactSATrace
}

function CircularCompactPSARTSATrace(;
    capacity,
    priority_type=Float32,
    reward_type = Float32,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
    kw...
)
    CombinedTrace(
        Trace(
            reward=CircularArrayBuffer{reward_type}(reward_size..., capacity),
            terminal=CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
            priority=SumTree(priority_type, capacity)
        ),
        CircularCompactSATrace(;capacity=capacity, kw...),
    )
end

#####
# CircularCompactPSALRTSALTrace
#####

const CircularCompactPSALRTSALTrace = CombinedTrace{
    <:Trace{<:NamedTuple{(:reward, :terminal,:priority), <:Tuple{<:CircularArrayBuffer, <:CircularArrayBuffer, <:SumTree}}},
    <:CircularCompactSALTrace
}

function CircularCompactPSALRTSALTrace(;
    capacity,
    priority_type=Float32,
    reward_type = Float32,
    reward_size = (),
    terminal_type = Bool,
    terminal_size = (),
    kw...
)
    CombinedTrace(
        Trace(
            reward=CircularArrayBuffer{reward_type}(reward_size..., capacity),
            terminal=CircularArrayBuffer{terminal_type}(terminal_size..., capacity),
            priority=SumTree(priority_type, capacity)
        ),
        CircularCompactSALTrace(;capacity=capacity, kw...),
    )
end