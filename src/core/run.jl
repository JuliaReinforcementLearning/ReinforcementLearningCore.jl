import Base: run

run(agent::AbstractAgent, env::AbstractEnv, stop_condition, hook::AbstractHook=EmptyHook()) = run(DynamicStyle(agent), agent, env, stop_condition, hook)

function run(::Sequential, agent::AbstractAgent, env::AbstractEnv, stop_condition, hook::AbstractHook)

    reset!(env)
    obs = observe(env)
    agent(PRE_EPISODE_STAGE, obs)
    hook(PRE_EPISODE_STAGE, agent, env, obs)

    while true
        action = agent(PRE_ACT_STAGE, obs)
        hook(PRE_ACT_STAGE, agent, env, ObsAndAction(obs, action))

        env(action)
        obs = observe(env)

        agent(POST_ACT_STAGE, obs)
        hook(POST_ACT_STAGE, agent, env, obs)

        if is_terminal(obs)
            agent(POST_EPISODE_STAGE, obs)
            hook(POST_EPISODE_STAGE, agent, env, obs)
            stop_condition(agent, env, obs) && break

            reset!(env)
            obs = observe(env)
            agent(PRE_EPISODE_STAGE, obs)
            hook(PRE_EPISODE_STAGE, agent, env, obs)
        else
            stop_condition(agent, env, obs) && break
        end
    end
    # !!! allow the agent to see the last observation, please!
    agent(PRE_ACT_STAGE, obs)
end
