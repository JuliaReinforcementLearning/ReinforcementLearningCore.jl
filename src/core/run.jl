import Base: run

run(agent, env::AbstractEnv, args...) = run(DynamicStyle(env), agent, env, args...)

function run(
    ::Sequential,
    agent::AbstractAgent,
    env::AbstractEnv,
    stop_condition,
    hook::AbstractHook = EmptyHook(),
)

    reset!(env)
    obs = observe(env)
    agent(PRE_EPISODE_STAGE, obs)
    hook(PRE_EPISODE_STAGE, agent, env, obs)
    action = agent(PRE_ACT_STAGE, obs)
    hook(PRE_ACT_STAGE, agent, env, obs, action)

    while true
        env(action)
        obs = observe(env)
        agent(POST_ACT_STAGE, obs)
        hook(POST_ACT_STAGE, agent, env, obs)

        if get_terminal(obs)
            agent(POST_EPISODE_STAGE, obs)  # let the agent see the last observation
            hook(POST_EPISODE_STAGE, agent, env, obs)

            stop_condition(agent, env, obs) && break

            reset!(env)
            obs = observe(env)
            agent(PRE_EPISODE_STAGE, obs)
            hook(PRE_EPISODE_STAGE, agent, env, obs)
            action = agent(PRE_ACT_STAGE, obs)
            hook(PRE_ACT_STAGE, agent, env, obs, action)
        else
            stop_condition(agent, env, obs) && break
            action = agent(PRE_ACT_STAGE, obs)
            hook(PRE_ACT_STAGE, agent, env, obs, action)
        end
    end
    hook
end

function run(
    ::Sequential,
    agent::AbstractAgent,
    env::MultiThreadEnv,
    stop_condition,
    hook::AbstractHook = EmptyHook(),
)

    while true
        reset!(env)
        obs = observe(env)
        action = agent(PRE_ACT_STAGE, obs)
        hook(PRE_ACT_STAGE, agent, env, obs, action)

        env(action)
        obs = observe(env)
        agent(POST_ACT_STAGE, obs)
        hook(POST_ACT_STAGE, agent, env, obs)

        if stop_condition(agent, env, obs)
            agent(PRE_ACT_STAGE, obs)  # let the agent see the last observation
            break
        end
    end
    hook
end

function run(
    ::Sequential,
    agents::Tuple{Vararg{<:AbstractAgent}},
    env::AbstractEnv,
    stop_condition,
    hooks = [EmptyHook() for _ in agents],
)
    reset!(env)
    observations = [observe(env, get_role(agent)) for agent in agents]

    valid_action = rand(get_action_space(env))  # init with a dummy value

    # async here?
    for (agent, obs, hook) in zip(agents, observations, hooks)
        agent(PRE_EPISODE_STAGE, obs)
        hook(PRE_EPISODE_STAGE, agent, env, obs)
        action = agent(PRE_ACT_STAGE, obs)
        hook(PRE_ACT_STAGE, agent, env, obs, action)
        # for Sequential environments, only one action is valid
        if get_current_player(env) == get_role(agent)
            valid_action = action
        end
    end

    while true
        env(valid_action)

        observations = [observe(env, get_role(agent)) for agent in agents]

        for (agent, obs, hook) in zip(agents, observations, hooks)
            agent(POST_ACT_STAGE, obs)
            hook(POST_ACT_STAGE, agent, env, obs)
        end

        if get_terminal(observations[1])
            for (agent, obs, hook) in zip(agents, observations, hooks)
                agent(POST_EPISODE_STAGE, obs)
                hook(POST_EPISODE_STAGE, agent, env, obs)
            end

            stop_condition(agents, env, observations) && break

            reset!(env)

            observations = [observe(env, get_role(agent)) for agent in agents]

            # async here?
            for (agent, obs, hook) in zip(agents, observations, hooks)
                agent(PRE_EPISODE_STAGE, obs)
                hook(PRE_EPISODE_STAGE, agent, env, obs)
                action = agent(PRE_ACT_STAGE, obs)
                hook(PRE_ACT_STAGE, agent, env, obs, action)
                # for Sequential environments, only one action is valid
                if get_current_player(env) == get_role(agent)
                    valid_action = action
                end
            end
        else
            stop_condition(agents, env, observations) && break
            for (agent, obs, hook) in zip(agents, observations, hooks)
                action = agent(PRE_ACT_STAGE, obs)
                hook(PRE_ACT_STAGE, agent, env, obs, action)
                # for Sequential environments, only one action is valid
                if get_current_player(env) == get_role(agent)
                    valid_action = action
                end
            end
        end
    end
    hooks
end

function run(
    ::Simultaneous,
    agents::Tuple{Vararg{<:AbstractAgent}},
    env::AbstractEnv,
    stop_condition,
    hook = EmptyHook(),
)
    reset!(env)
    observations = [observe(env, get_role(agent)) for agent in agents]
    hook(PRE_EPISODE_STAGE, agents, env, observations)
    actions = [agent(obs) for obs in observations]
    hook(PRE_ACT_STAGE, agents, env, observations, actions)

    while true
        env(actions)

        for (i, agent) in enumerate(agents)
            observations[i] = observe(env, get_role(agent))
        end
        hook(POST_ACT_STAGE, agents, env, observations)

        if get_terminal(observations[1])
            for (agent, obs) in zip(agents, observations)
                agent(POST_EPISODE_STAGE, obs)
            end
            hook(POST_EPISODE_STAGE, agents, env, observations)

            stop_condition(agents, env, observations) && break

            reset!(env)

            for (i, agent) in enumerate(agents)
                observations[i] = observe(env, get_role(agent))
            end
            hook(PRE_EPISODE_STAGE, agents, env, observations)
            for (i, agent) in enumerate(agents)
                actions[i] = agent(observations[i])
            end
            hook(PRE_ACT_STAGE, agents, env, observations, actions)
        else
            stop_condition(agents, env, observations) && break
            for (i, agent) in enumerate(agents)
                actions[i] = agent(observations[i])
            end
            hook(PRE_ACT_STAGE, agents, env, observations, actions)
        end
    end
    hook
end
