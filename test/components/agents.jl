@testset "Agent" begin
    env = CartPoleEnv()
    s = state(env)
    agent = Agent(
        policy = RandomPolicy(action_space(env)),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10,
            state = eltype(s) => size(s),
        ),
    )

    @testset "trajectory" begin
        length(t::AbstractTrajectory) = [size(t[k], ndims(t[k])) for k in keys(t)]

        update!(agent.policy, agent.trajectory, env, PRE_EPISODE_STAGE)
        @test length(agent.trajectory) == [0, 0, 0, 0]

        update!(agent.trajectory, agent.policy, env, PRE_ACT_STAGE)
        @test length(agent.trajectory) == [1, 1, 0, 0]

        update!(agent.trajectory, agent.policy, env, POST_ACT_STAGE)
        @test length(agent.trajectory) == [1, 1, 1, 1]

        update!(agent.trajectory, agent.policy, env, POST_EPISODE_STAGE)
        @test length(agent.trajectory) == [2, 2, 1, 1]
    end
end
