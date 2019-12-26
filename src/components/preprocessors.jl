export CopyStateProcessor

using MacroTools:@forward

struct CacheStateObs{O, S}
    obs::O
    state::S
end

@forward CacheStateObs.obs RLBase.ActionStyle, RLBase.legal_actions, RLBase.legal_actions_mask, RLBase.is_terminal, RLBase.get_reward

RLBase.get_state(obs::CacheStateObs) = obs.state

#####
# CopyStateProcessor
#####

struct CopyStateProcessor <: AbstractPreprocessor end

(p::CopyStateProcessor)(obs) = CacheStateObs(obs, deepcopy(get_state(obs)))