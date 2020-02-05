export TabularApproximator

"""
    TabularApproximator(table<:AbstractArray)

For `table` of 1-d, it will create a [`VApproximator`](@ref). For `table` of 2-d, it will create a [`QApproximator`].

!!! warning
    For `table` of 2-d, the first dimension is action and the second dimension is state.
"""
struct TabularApproximator{N, T<:AbstractArray} <: AbstractApproximator
    table::T
    function TabularApproximator(table::T) where {T<:AbstractArray}
        n = ndims(table)
        n <= 2 || throw(ArgumentError("the dimention of table must be <= 2"))
        new{n,T}(table)
    end
end

function TabularApproximator(;n_state, n_action=nothing, init=0.)
    table = isnothing(n_action) ? fill(init, n_state) : fill(init, n_action, n_state)
    TabularApproximator(table)
end

(app::TabularApproximator{1})(s) = @views app.table[s]

(app::TabularApproximator{2})(s, a) = @views app.table[a, s]

function RLBase.update!(app::TabularApproximator, correction::Pair)
    s, e = correction
    app.table[s] += e
end

RLBase.ApproximatorStyle(::TabularApproximator{1}) = VApproximator()
RLBase.ApproximatorStyle(::TabularApproximator{2}) = QApproximator()