export CloneStateProcessor

#####
# CloneStateProcessor
#####

struct CloneStateProcessor <: AbstractPreprocessor end

(p::CloneStateProcessor)(obs) = StateOverriddenObs(obs, deepcopy(get_state(obs)))