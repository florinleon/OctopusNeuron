import os
import config
from runner import run_experiment


# ----- The experiment to run ------------------------------------------------------------------------------------------

# The lines below repeat values of config.py, so this runs the defaults: class-incremental MNIST with task context,
# learned from reward by recirculation. Results go to results/<ExperimentName>/seed<N>/, one folder per run, with a
# summary over the runs in results/<ExperimentName>/.
config.ExperimentName = "mnist_recirculation"
config.NumExperiments = 1


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_experiment()
