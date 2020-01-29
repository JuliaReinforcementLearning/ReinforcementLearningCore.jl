using ReinforcementLearningCore
using Random
using Test
using StatsBase
using Distributions

@testset "ReinforcementLearningCore.jl" begin
    include("core/core.jl")
    include("components/components.jl")
    include("utils/utils.jl")
end
