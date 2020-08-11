export AbstractTrace

"""
    AbstractTrace

A trace is used to record some useful information
during the interactions between agents and environments.

Required Methods:

- `Base.haskey(t::AbstractTrace, s::Symbol)`
- `Base.getproperty(t::AbstractTrace, s::Symbol)`
- `Base.keys(t::AbstractTrace)`
- `Base.push!(t::AbstractTrace, kv::Pair{Symbol})`
- `Base.pop!(t::AbstractTrace, s::Symbol)`

Optional Methods:

- `isfull`
- `empty!`

"""
abstract type AbstractTrace end

function Base.push!(t::AbstractTrace;kwargs...)
    for kv in kwargs
        push!(t, kv)
    end
end

"""
    Base.pop!(t::AbstractTrace, s::Symbol...)

`pop!` out one element of the traces specified in `s`
"""
function Base.pop!(t::AbstractTrace, s::Tuple{Vararg{Symbol}})
    NamedTuple{s}(pop!(t, x) for x in s)
end

Base.pop!(t::AbstractTrace) = pop!(t, keys(t))

function Base.empty!(t::AbstractTrace)
    for s in keys(t)
        empty!(t[s])
    end
end