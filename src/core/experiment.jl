export @experiment_str

using BSON

Base.@kwdef mutable struct Experiment
    agent
    env
    stop_condition
    hook
    description::String
end

function RLBase.save(dir::String, experiment::Experiment)
    mkpath(dir)
    BSON.bson(
        joinpath(dir, "experiment_meta.bson"),
        Dict(
            :agent_type => typeof(experiment.agent),
            :env_type => typeof(experiment.env),
            :stop_condition => experiment.stop_condition,
            :hook => experiment.hook,
            :description => experiment.description
        ))
    RLBase.save(dir, experiment.agent)
    RLBase.save(joinpath(dir, "env.bson"), experiment.env)
end

function RLBase.load(dir::String, ::Type{<:Experiment})
    @info "loading experiment from $s"
    BSON.@load joinpath(dir, "experiment_meta.bson") agent_type env_type stop_condition hook description
    agent = RLBase.load(dir, agent_type)
    env = RLBase.load(joinpath(dir, "env.bson"), env_type)
    Experiment(agent, env, stop_condition, hook, description)
end

macro experiment_str(s)
    Experiment(s)
end

function Experiment(s::String)
    if isdir(s)
        RLBase.load(s, Experiment)
    else
        m = match(r"(?<source>\w+)_(?<method>\w+)_(?<env>\w+)(\((?<game>\w*)\))?", s)
        isnothing(m) && throw(ArgumentError("invalid format, got $s, expected format is a local dir or a predefined experiment like dopamine_dqn_atari(pong)`"))
        Experiment(Val(Symbol(m[:source])), Val(Symbol(m[:method])), Val(Symbol(m[:env])), m[:game])
    end
end

Base.run(x::Experiment) = run(x.agent, x.env, x.stop_condition, x.hook)
