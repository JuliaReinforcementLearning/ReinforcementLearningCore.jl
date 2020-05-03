export AbstractApproximator

"""
    (app::AbstractApproximator)(obs)

An approximator is a functional object for value estimation.
It serves as a black box to provides an abstraction over different 
kinds of approximate methods (for example DNN provided by Flux or Knet).
"""
abstract type AbstractApproximator end

"""
    update!(a::AbstractApproximator, correction)

Usually the `correction` is the gradient of inner parameters.
"""
function update!(a::AbstractApproximator, correction) end