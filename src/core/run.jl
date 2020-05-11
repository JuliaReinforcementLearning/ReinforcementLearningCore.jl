export @experiment_str
import Base: run

mutable struct Experiment
    agent
    env
    stop_condition
    hook
end

macro experiment_str(s)
    Experiment(s)
end

function Experiment(s::String)
    m = match(r"(?<source>\w+)_(?<method>\w+)_(?<env>\w+)(\((?<game>\w*)\))?", s)
    isnothing(m) && throw(ArgumentError("invalid format, got $s, expected format is like dopamine_dqn_atari(pong)"))
    Experiment(Val(Symbol(m[:source])), Val(Symbol(m[:method])), Val(Symbol(m[:env])), m[:game])
end

run(x::Experiment) = run(x.agent, x.env, x.stop_condition, x.hook)

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
