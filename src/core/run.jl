import Base: run

run(agent, env::AbstractEnv, args...) =
    run(DynamicStyle(env), NumAgentStyle(env), agent, env, args...)

function run(
    ::Sequential,
    ::SingleAgent,
    agent::AbstractAgent,
    env::AbstractEnv,
    stop_condition,
    hook::AbstractHook = EmptyHook(),
)

    reset!(env)
    agent(PRE_EPISODE_STAGE, env)
    hook(PRE_EPISODE_STAGE, agent, env)
    action = agent(PRE_ACT_STAGE, env)
    hook(PRE_ACT_STAGE, agent, env, action)

    while true
        env(action)
        agent(POST_ACT_STAGE, env)
        hook(POST_ACT_STAGE, agent, env)

        if get_terminal(env)
            agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
            hook(POST_EPISODE_STAGE, agent, env)

            stop_condition(agent, env) && break

            reset!(env)
            agent(PRE_EPISODE_STAGE, env)
            hook(PRE_EPISODE_STAGE, agent, env)
            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)
        else
            stop_condition(agent, env) && break
            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)
        end
    end
    hook
end

function run(
    ::Sequential,
    ::SingleAgent,
    agent::AbstractAgent,
    env::MultiThreadEnv,
    stop_condition,
    hook::AbstractHook = EmptyHook(),
)

    while true
        reset!(env)
        action = agent(PRE_ACT_STAGE, env)
        hook(PRE_ACT_STAGE, agent, env, action)

        env(action)
        agent(POST_ACT_STAGE, env)
        hook(POST_ACT_STAGE, agent, env)

        if stop_condition(agent, env)
            agent(PRE_ACT_STAGE, env)  # let the agent see the last observation
            break
        end
    end
    hook
end

function run(
    ::Sequential,
    ::MultiAgent,
    agents::Tuple{Vararg{<:AbstractAgent}},
    env::AbstractEnv,
    stop_condition,
    hooks = [EmptyHook() for _ in agents],
)
    @assert length(agents) == get_num_players(env)

    hooks = Dict(get_role(agent) => hook for (agent, hook) in zip(agents, hooks))
    agents = Dict(get_role(agent) => agent for agent in agents)
    reset!(env)
    
    agent = agents[get_current_player(env)]
    hook = hooks[get_current_player(env)]

    for (A,H) in zip(values(agents), values(hooks))
        A(PRE_EPISODE_STAGE, env)
        H(PRE_EPISODE_STAGE, A, env)
    end

    action = agent(PRE_ACT_STAGE, env)
    hook(PRE_ACT_STAGE, agent, env, action)

    while true
        env(action)
        agent(POST_ACT_STAGE, env)
        hook(POST_ACT_STAGE, agent, env)

        if get_terminal(env)
            for (A,H) in zip(values(agents), values(hooks))
                A(POST_EPISODE_STAGE, env)
                H(POST_EPISODE_STAGE, A, env)
            end

            stop_condition(agent, env) && break

            reset!(env)

            for (A,H) in zip(values(agents), values(hooks))
                A(PRE_EPISODE_STAGE, env)
                H(PRE_EPISODE_STAGE, A, env)
            end

            agent = agents[get_current_player(env)]
            hook = hooks[get_current_player(env)]
            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)
        else
            stop_condition(agent, env) && break

            agent = agents[get_current_player(env)]
            hook = hooks[get_current_player(env)]
            action = agent(PRE_ACT_STAGE, env)
            hook(PRE_ACT_STAGE, agent, env, action)
        end
    end

    hooks
end
