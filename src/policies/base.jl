export AbstractStage,
    PreExperimentStage,
    PostExperimentStage,
    PreEpisodeStage,
    PostEpisodeStage,
    PreActStage,
    PostActStage,
    PRE_EXPERIMENT_STAGE,
    POST_EXPERIMENT_STAGE,
    PRE_EPISODE_STAGE,
    POST_EPISODE_STAGE,
    PRE_ACT_STAGE,
    POST_ACT_STAGE,
    set_mode!,
    AbstractMode,
    TrainMode,
    TRAIN_MODE,
    TestMode,
    TEST_MODE,
    EvalMode,
    EVAL_MODE

#####
# Stage
#####

abstract type AbstractStage end

struct PreExperimentStage <: AbstractStage end
const PRE_EXPERIMENT_STAGE = PreExperimentStage()

struct PostExperimentStage <: AbstractStage end
const POST_EXPERIMENT_STAGE = PostExperimentStage()

struct PreEpisodeStage <: AbstractStage end
const PRE_EPISODE_STAGE = PreEpisodeStage()

struct PostEpisodeStage <: AbstractStage end
const POST_EPISODE_STAGE = PostEpisodeStage()

struct PreActStage <: AbstractStage end
const PRE_ACT_STAGE = PreActStage()

struct PostActStage <: AbstractStage end
const POST_ACT_STAGE = PostActStage()

#####
# Modes
#####

abstract type AbstractMode end

struct TrainMode <: AbstractMode end
const TRAIN_MODE = TrainMode()

struct EvalMode <: AbstractMode end
const EVAL_MODE = EvalMode()

struct TestMode <: AbstractMode end
const TEST_MODE = TestMode()

function set_mode!(p, ::TrainMode)
    for x in Flux.trainable(p)
        Flux.trainmode!(x)
    end
end

function set_mode!(p, ::Union{TestMode, EvalMode})
    for x in Flux.trainable(p)
        Flux.testmode!(x)
    end
end