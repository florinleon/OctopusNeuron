import os
import config
from runner import run_experiment


# ----- Settings shared by the Iris experiments ------------------------------------------------------------------------

# One layer of 64 neurons (a 4 x 4 grid with 4 neurons per position), each sampling 12 observation units anywhere; the
# readout reads their trigger outputs. Four features with nearly linear class boundaries need far less than the two
# layers of 392 neurons that config.py sets up for MNIST, and a small network keeps every run short.
IrisLayers = [
    {
        "name": "layer1",
        "grid": [4, 4],
        "neurons_per_position": 4,
        "inputs": [
            {"source": "observation", "kind": "global", "fan_in": 12},
        ],
    },
]


def iris_settings():
    """Switch to the Iris stream and adapt the MNIST-sized defaults of config.py to a dataset of 150 flowers.

    Every Iris experiment below calls this first and then sets only what distinguishes it.
    """
    config.ProblemName = "iris"
    config.Layers = IrisLayers
    config.ReadoutSources = ["layer1.z"]
    # A task holds 70 or 105 training flowers (35 per class with the default split). With batches of 16, 300 steps show
    # every flower about 45 to 70 times.
    config.TrainingBatchSize = 16
    config.StepsPerTask = 300
    # 25 steps are 400 flowers, four to six passes over a task: long enough to average out the noise of single batches,
    # short enough to measure how quickly a task is picked up rather than the reward finally reached.
    config.SpeedWindow = 25
    config.LogEvery = 50
    config.EvaluationBatchSize = 50
    # One test flower moves the accuracy on a class of 15 by almost 7 points, so the summary averages over 10 runs, each
    # with its own split of the flowers.
    config.NumExperiments = 1
    # Without a task code the same neurons learn every task, so the final task (all three classes) has to consolidate
    # what the earlier ones left in them. The *_context experiments below give every task its own gating code instead.
    config.ProvideContext = False


# ----- The experiment to run ------------------------------------------------------------------------------------------

# Iris in three stages: setosa vs versicolor, then versicolor vs virginica with no setosa at all (the accuracy on the
# first task afterwards measures forgetting), then all three classes. Results go to results/<ExperimentName>/seed<N>/,
# one folder per run, with a summary over the runs in results/<ExperimentName>/.
iris_settings()
config.ExperimentName = "iris_three_stage"
#config.IrisTasks = [[0, 1], [1, 2], [0, 1, 2]]  #1
#config.IrisTasks = [[1, 2], [0, 1], [0, 2]]  #2
config.IrisTasks = [[0, 1], [1, 2]]  #3


# ----- Other experiments: to run one, put its lines in place of the block above ---------------------------------------

# Reference for every stream: all three classes as a single task, that is ordinary joint training.
# iris_settings()
# config.ExperimentName = "iris_joint"
# config.IrisTasks = [[0, 1, 2]]

# The three stages with a one-hot task code, which gates about half of the branches of every neuron, chosen at random
# for each task.
# iris_settings()
# config.ExperimentName = "iris_three_stage_context"
# config.ProvideContext = True

# Small vs medium, then all three classes. The second task shows setosa and versicolor again, so this measures how
# readily a confident policy makes room for virginica more than how much it forgets: compare learning_speed of the
# second task in results.json with that of the single task of iris_joint.
# iris_settings()
# config.ExperimentName = "iris_curriculum"
# config.IrisTasks = [[0, 1], [0, 1, 2]]

# Strictly class-incremental, like the MNIST stream: virginica alone after setosa vs versicolor, with task code. Within
# the second task answering virginica to every flower earns full reward, so only gating and consolidation can keep the
# first two classes.
# iris_settings()
# config.ExperimentName = "iris_strict_incremental"
# config.IrisTasks = [[0, 1], [2]]
# config.ProvideContext = True

# The three stages learned by backpropagation on the labels: how far exact credit assignment gets with the same network.
# iris_settings()
# config.ExperimentName = "iris_gradient_supervised"
# config.LearningMethod = "gradient"
# config.GradientObjective = "supervised"

# The three stages with a dense input code, in which most units are active for most flowers, so all tasks act largely
# on the same synapses. With only 8 observation units every neuron samples all of them, and its 4 remaining contacts
# repeat some of them.
# iris_settings()
# config.ExperimentName = "iris_complement_code"
# config.IrisEncoding = "complement"

# Class-incremental MNIST with task context, learned from reward by recirculation: the defaults of config.py.
# config.ExperimentName = "mnist_recirculation"
# config.NumExperiments = 1


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_experiment()
