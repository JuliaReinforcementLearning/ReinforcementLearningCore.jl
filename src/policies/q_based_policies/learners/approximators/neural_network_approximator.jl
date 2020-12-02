export NeuralNetworkApproximator, ActorCritic

using Flux
import Functors: functor

"""
    NeuralNetworkApproximator(;kwargs)

Use a DNN model for value estimation.

# Keyword arguments

- `model`, a Flux based DNN model.
- `optimizer=nothing`
"""
Base.@kwdef struct NeuralNetworkApproximator{M,O} <: AbstractApproximator
    model::M
    optimizer::O = nothing
end

function (app::NeuralNetworkApproximator)(x)
    app.model(send_to_device(device(app.model), x))
end

functor(x::NeuralNetworkApproximator) =
    (model = x.model,), y -> NeuralNetworkApproximator(y.model, x.optimizer)

device(app::NeuralNetworkApproximator) = device(app.model)

RLBase.update!(app::NeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, params(app), gs)

Base.copyto!(dest::NeuralNetworkApproximator, src::NeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, params(src))

#####
# ActorCritic
#####

"""
    ActorCritic(;actor, critic, optimizer=ADAM())

The `actor` part must return logits (*Do not use softmax in the last layer!*), and the `critic` part must return a state value.
"""
Base.@kwdef struct ActorCritic{A,C,O} <: AbstractApproximator
    actor::A
    critic::C
    optimizer::O = ADAM()
end

functor(x::ActorCritic) =
    (actor = x.actor, critic = x.critic), y -> ActorCritic(y.actor, y.critic, x.optimizer)

RLBase.update!(app::ActorCritic, gs) = Flux.Optimise.update!(app.optimizer, params(app), gs)

actor(ac::ActorCritic) = x -> ac.actor(send_to_device(device(ac.actor), x))
critic(ac::ActorCritic) = x -> ac.critic(send_to_device(device(ac.actor), x))
