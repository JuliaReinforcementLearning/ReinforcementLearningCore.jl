export NStepInserter, UniformBatchSampler

using Random

#####
# Inserters
#####

abstract type AbstractInserter end

Base.@kwdef struct NStepInserter <: AbstractInserter
    n::Int = 1
end

function Base.push!(
    t::CircularSARTSATrajectory,
    𝕥::CircularCompactSARTSATrajectory,
    adder::NStepInserter,
)
    N = length(𝕥[:terminal])
    n = adder.n
    for i in 1:(N-n+1)
        push!(
            t;
            state = select_last_dim(𝕥[:state], i),
            action = select_last_dim(𝕥[:action], i),
            reward = select_last_dim(𝕥[:reward], i),
            terminal = select_last_dim(𝕥[:terminal], i),
            next_state = select_last_dim(𝕥[:next_state], i + n - 1),
            next_action = select_last_dim(𝕥[:next_action], i + n - 1),
        )
    end
end

#####
# Samplers
#####

abstract type AbstractSampler end

struct UniformBatchSampler <: AbstractSampler
    batch_size::Int
end

StatsBase.sample(t::AbstractTrajectory, sampler::AbstractSampler) =
    sample(Random.GLOBAL_RNG, t, sampler)

function StatsBase.sample(
    rng::AbstractRNG,
    t::VectSARTSATrajectory,
    sampler::UniformBatchSampler,
    trace_names=(:state, :action, :reward, :terminal, :next_state, :next_action)
)
    inds = rand(rng, 1:length(t), sampler.batch_size)
    NamedTuple{trace_names}(Flux.batch(view(t[x], inds)) for x in trace_names)
end

function StatsBase.sample(
    rng::AbstractRNG,
    t::Union{CircularCompactSARTSATrajectory, CircularSARTSATrajectory},
    sampler::UniformBatchSampler,
    trace_names=(:state, :action, :reward, :terminal, :next_state, :next_action)
)
    inds = rand(rng, 1:length(t), sampler.batch_size)
    NamedTuple{trace_names}(convert(Array, consecutive_view(t[x], inds)) for x in trace_names)
end
