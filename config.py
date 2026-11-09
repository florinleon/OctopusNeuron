# ----- Experiment -----------------------------------------------------------------------------------------------------

ExperimentName = "mnist_recirculation"
# The same experiment is run NumExperiments times, with the seeds InitialSeed, InitialSeed + 1, ... Each run is written
# to <ResultsDir>/<ExperimentName>/seed<N>/, and a summary over all runs (mean and standard deviation) to
# <ResultsDir>/<ExperimentName>/.
NumExperiments = 10
InitialSeed = 10
ResultsDir = "results"
# Floating point type of all model arrays; float32 halves memory traffic and is accurate enough for the tree solver.
Dtype = "float32"


# ----- Problem --------------------------------------------------------------------------------------------------------

ProblemName = "mnist"
# Which IDX dataset to download: mnist or fashion_mnist (same format, same sizes).
Dataset = "mnist"
DataDir = "data"
# class_incremental: each task brings new digits (ClassesPerTask of them), all ten actions stay available.
# permuted: every task shows all ten digits under its own fixed random pixel permutation.
ProblemMode = "class_incremental"
ClassesPerTask = 2
# Order in which the classes are split into consecutive tasks (class_incremental only).
ClassOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
NumPermutations = 5
# When true, every observation comes with a one-hot task code that reaches the neurons as branch-gating inhibition.
ProvideContext = True


# ----- Training -------------------------------------------------------------------------------------------------------

# Number of independent streams processed in parallel; learning rules average their updates over the batch.
TrainingBatchSize = 32
# Environment steps per task; with TrainingBatchSize 32 each step answers 32 images.
StepsPerTask = 500
LogEvery = 250
# Window (in steps) at the start of each task used to measure how fast a new task is picked up.
SpeedWindow = 300


# ----- Evaluation -----------------------------------------------------------------------------------------------------

EvaluationBatchSize = 500
# 0 evaluates on the whole test split of a task; a positive value caps the number of test episodes.
EvaluationMaxEpisodes = 0


# ----- Model: morphology ----------------------------------------------------------------------------------------------

# star: one trunk segment (the trigger) with identical unbranched branches; custom: explicit tree below.
MorphologyType = "star"
Branches = 4
SegmentsPerBranch = 2
TrunkLeak = 2.0
BranchLeak = 0.5
TrunkCoupling = 2.0
BranchCoupling = 2.0
# all_branch: every branch segment carries local output synapses; tips: only the last segment of each branch.
OutputPlacement = "all_branch"
# Used only when MorphologyType is custom: CustomParents[k] is the segment closer to the trigger (-1 for the trigger
# itself).
CustomParents = []
CustomCouplings = []
CustomLeaks = []
CustomOutputSegments = []
CustomGateSegments = []


# ----- Model: neuron --------------------------------------------------------------------------------------------------

# Value an excitatory channel pulls its segment towards, and the value an inhibitory channel pulls towards.
ReversalExcitatory = 1.0
ReversalInhibitory = 0.0
# Each presynaptic source gets this many excitatory and inhibitory synapses on every neuron that samples it.
ExcitatoryPerSource = 1
InhibitoryPerSource = 1
# any: synapses may land on any segment including the trunk; branches: only on branch segments.
SynapsePlacement = "any"
# Firing probability is sigmoid((v_trigger - threshold) / T) with T = FiringTemperature * spread of trigger values.
# Trigger outputs are firing probabilities; the second pass is a response around this deterministic forward state.
FiringTemperature = 0.3
# Strength of the back-boost: a current proportional to the neuron's output z, injected at the trigger.
BackBoost = 2.0
WeightMax = 10.0


# ----- Model: context -------------------------------------------------------------------------------------------------

# Each context unit inhibits the entry segment of a random subset of every neuron's branches (ContextGateFraction of
# them) and the trunk of a random subset of the neurons (ContextTrunkGateFraction of them).
ContextGateFraction = 0.5
ContextTrunkGateFraction = 0.0
ContextStrength = 6.0


# ----- Model: calibration ---------------------------------------------------------------------------------------------

# Synaptic weights are rescaled so the mean synaptic conductance per segment equals ratio * mean leak.
ConductanceRatio = 1.0
# Trigger thresholds start at the value that makes each neuron's firing probability exceed 0.5 on this fraction of
# calibration inputs.
CalibrationTargetRate = 0.1
# Local output thresholds start at the value that makes outputs active on this fraction of inputs.
OutputActiveFraction = 0.5
CalibrationBatchSize = 256


# ----- Model: layers --------------------------------------------------------------------------------------------------

# Each layer is a grid of positions in the unit square with several neurons per position.
# An input block draws fan_in sources per neuron, either near the neuron (local) or anywhere (global).
# Source names: observation, <layer>.y (local outputs) and <layer>.z (trigger outputs).
Layers = [
    {
        "name": "layer1",
        "grid": [7, 7],
        "neurons_per_position": 8,
        "inputs": [
            {"source": "observation", "kind": "local", "radius": 0.15, "fan_in": 64},
        ],
    },
    {
        "name": "layer2",
        "grid": [7, 7],
        "neurons_per_position": 8,
        "inputs": [
            {"source": "layer1.y", "kind": "local", "radius": 0.2, "fan_in": 96},
            {"source": "layer1.z", "kind": "global", "fan_in": 32},
        ],
    },
]


# ----- Model: readout -------------------------------------------------------------------------------------------------

# Only long-distance (trigger) outputs reach the readout.
ReadoutSources = ["layer2.z"]
# Per-action biases are shared by every input: rewarding the classes of a new task raises their bias for the
# images of old tasks as well, so on class-incremental streams they are a direct path to forgetting.
ReadoutBias = False


# ----- Learning -------------------------------------------------------------------------------------------------------

# recirculation: two passes per step, every synapse learns from how its own segment moved between them;
# gradient: exact gradients through the whole network (a reference for recirculation).
LearningMethod = "recirculation"


# ----- Learning: settings shared by recirculation and gradient --------------------------------------------------------

# Discount of future value in multi-step episodes (MNIST episodes have a single step, so it has no effect there).
Gamma = 0.9
# Softmax temperature over the predicted rewards and share of uniformly random actions (gradient: reward-only
# objectives only).
ActionTemperature = 0.1
Exploration = 0.1


# ----- Learning: recirculation ----------------------------------------------------------------------------------------

# Learning rate of every synapse in every layer: dw = rate * x * (E - v_segment) * shift_segment. Calibrated weights
# are around 0.3; larger rates learn each task faster but also overwrite earlier tasks faster.
SynapseRate = 20.0
# Normalised step size of the readout (stable below 2); only the chosen action's prediction learns.
ValueRate = 0.5
# Synapses that changed a lot become less plastic: rate / (1 + strength * importance); importance decays slowly.
# This is the main balance between keeping old tasks (larger) and learning new ones quickly (smaller).
ConsolidationStrength = 10000.0
ConsolidationDecay = 0.0002


# ----- Learning: gradient ---------------------------------------------------------------------------------------------

# supervised: cross-entropy on labels (needs info["target"]).
# action_values / actor_critic: reward only, with the same readout rule as the recirculation learner (action_values)
# or a policy-gradient actor with a critic (actor_critic), by backprop.
GradientObjective = "supervised"
GradientLearningRate = 0.001
AdamBeta1 = 0.9
AdamBeta2 = 0.999
# Weights of the critic loss and of the entropy bonus (actor_critic objective only).
CriticLossWeight = 0.5
EntropyWeight = 0.01
# Penalty on the mean trigger output of every layer, which keeps long-distance activity sparse.
SparsityWeight = 0.001
LearnThresholds = True
