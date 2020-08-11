@testset "traces" begin
    @testset "Trace" begin
        t = Trace(;state=Vector{Int}(), reward=Vector{Bool}())
        @test (:state, :reward) == keys(t)
        @test haskey(t, :state)
        @test haskey(t, :reward)
        push!(t; state=3, reward=true)
        push!(t; state=4, reward=false)
        @test t[:state] == [3,4]
        @test t[:reward] == [true,false]
        pop!(t)
        @test t[:state] == [3]
        @test t[:reward] == [true]
        empty!(t)
        @test t[:state] == Int[]
        @test t[:reward] == Bool[]
    end

    @testset "SharedTrace" begin
        t = SharedTrace(Int[], :state)
        @test (:state, :next_state, :full_state) == keys(t)
        @test haskey(t, :state)
        @test haskey(t, :next_state)
        @test haskey(t, :full_state)
        @test t[:state] == Int[]
        @test t[:next_state] == Int[]
        @test t[:full_state] == Int[]
        push!(t;state=1,next_state=2)
        @test t[:state] == [1]
        @test t[:next_state] == [2]
        @test t[:full_state] == [1, 2]
        empty!(t)
        @test t[:state] == Int[]
        @test t[:next_state] == Int[]
        @test t[:full_state] == Int[]
    end

    @testset "EpisodicTrace" begin
        t = EpisodicTrace(
            Trace(;state=Vector{Int}(), reward=Vector{Bool}()),
            :reward
            )

        @test isfull(t) == false
        
        @test (:state, :reward) == keys(t)
        @test haskey(t, :state)
        @test haskey(t, :reward)
        push!(t; state=3, reward=true)

        @test isfull(t) == true

        push!(t; state=4, reward=false)
        @test t[:state] == [3,4]
        @test t[:reward] == [true,false]
        pop!(t)
        @test t[:state] == [3]
        @test t[:reward] == [true]
        empty!(t)
        @test t[:state] == Int[]
        @test t[:reward] == Bool[]
    end

    @testset "CombinedTrace" begin
        t = CircularCompactPSALRTSALTrace(;capacity=3, legal_actions_mask_size=(2,))
        push!(t; state=1, action=1, legal_actions_mask=[false, false])
        push!(t;reward=0.f0, terminal=false, priority=100, state=2, action=2, legal_actions_mask=[false, true])

        @test t[:state] == [1]
        @test t[:action] == [1]
        @test t[:legal_actions_mask] == [false false]'
        @test t[:reward] == [0.f0]
        @test t[:terminal] == [false]
        @test t[:priority] == [100]
        @test t[:next_state] == [2]
        @test t[:next_action] == [2]
        @test t[:next_legal_actions_mask] == [false true]'
        @test t[:full_state] == [1,2]
        @test t[:full_action] == [1,2]
        @test t[:full_legal_actions_mask] == [false false
        false true]

        push!(t;reward=1.f0, terminal=true, priority=200, state=3, action=3, legal_actions_mask=[true, true])

        @test t[:state] == [1,2]
        @test t[:action] == [1,2]
        @test t[:legal_actions_mask] == [false false
        false true]
        @test t[:reward] == [0.f0, 1.f0]
        @test t[:terminal] == [false, true]
        @test t[:priority] == [100,200]
        @test t[:next_state] == [2, 3]
        @test t[:next_action] == [2,3]
        @test t[:next_legal_actions_mask] == [false true 
        true true]
        @test t[:full_state] == [1,2,3]
        @test t[:full_action] == [1,2,3]
        @test t[:full_legal_actions_mask] == [false false true 
        false true true]

        pop!(t)

        @test t[:state] == [1]
        @test t[:action] == [1]
        @test t[:legal_actions_mask] == [false false]'
        @test t[:reward] == [0.f0]
        @test t[:terminal] == [false]
        @test t[:priority] == [100]
        @test t[:next_state] == [2]
        @test t[:next_action] == [2]
        @test t[:next_legal_actions_mask] == [false true]'
        @test t[:full_state] == [1,2]
        @test t[:full_action] == [1,2]
        @test t[:full_legal_actions_mask] == [false false
        false true]


        empty!(t)

        @test t[:state] == []
        @test t[:action] == []
        @test t[:reward] == []
        @test t[:terminal] == []
        @test t[:next_state] == []
        @test t[:next_action] == []
        @test t[:full_state] == []
        @test t[:full_action] == []
    end
end
