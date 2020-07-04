export TabularPolicy

using AbstractTrees

struct TabularPolicy{S,F,E} <: RLBase.AbstractPolicy
    probs::Dict{S,Vector{Float64}}
    key::F
    explorer::E
end

(p::TabularPolicy)(obs) = p.probs[p.key(obs)] |> p.explorer

RLBase.get_prob(p::TabularPolicy, obs) = p.probs[p.key(obs)]

function TabularPolicy(env::AbstractEnv;key=RLBase.get_state, explorer=WeightedExplorer(;is_normalized=true))
    k = key(observe(env))
    probs = Dict{typeof(k),Vector{Float64}}()
    for x in PreOrderDFS(env)
        if get_current_player(x) != get_chance_player(x)
            obs = observe(x)
            if !get_terminal(obs)
                legal_actions_mask = get_legal_actions_mask(obs)
                p = zeros(length(legal_actions_mask))
                p[legal_actions_mask] .= 1 / sum(legal_actions_mask)
                probs[key(obs)] = p
            end
        end
    end
    TabularPolicy(probs, key, explorer)
end
