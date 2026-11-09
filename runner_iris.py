"""Continual-learning experiment loop: build everything from the settings, train task after task, evaluate all tasks.

The runner is the only place where a problem, a network and a learner meet. It relies solely on the generic interfaces
(ProblemSpec, Environment, Learner), so a new problem only has to implement the problem interface and register itself
in Problems below to be usable with every model and learning method.

Protocol: the network is calibrated once on a batch of the first task's observations, paired with randomly chosen
context codes of all tasks (see calibration_contexts). Then, for every task in order, the learner
trains on that task's stream for StepsPerTask steps, after which every task (seen or not) is evaluated greedily on its
test set. The whole protocol is repeated NumExperiments times, with the seeds InitialSeed, InitialSeed + 1, ... Each
run is written to <ResultsDir>/<ExperimentName>/seed<N>/, and a summary over all runs to <ResultsDir>/<ExperimentName>/.

All settings live in config.py. main.py changes some of them before it calls run_experiment; the functions at the end
of this file list and check them.
"""

import os
import runpy
import time
import types

import numpy as np

import config
from gradient_learner import GradientLearner
from iris import IrisIncremental
from mnist import MnistIncremental
from network import Network
from recirculation_learner import RecirculationLearner
from records import ContinualRecord, summarize_runs


# Problems (task streams and their environments), selected by name in config.ProblemName.
Problems = {"mnist": MnistIncremental, "iris": IrisIncremental}

# Learning methods for unipolar networks, selected by name in config.LearningMethod.
Learners = {"recirculation": RecirculationLearner, "gradient": GradientLearner}

# Absolute path of config.py, from which the default value of every setting is read.
ConfigFile = os.path.abspath(config.__file__)


def make_problem(dtype, rng):
    if config.ProblemName not in Problems:
        raise ValueError("Unknown problem '" + str(config.ProblemName) + "'; known: " + ", ".join(sorted(Problems)))
    return Problems[config.ProblemName](dtype, rng)


def make_learner(network, rng):
    """Build the learner named by config.LearningMethod."""
    if config.LearningMethod not in Learners:
        raise ValueError("Unknown learning method '" + str(config.LearningMethod) + "'; known: "
                         + ", ".join(sorted(Learners)))
    return Learners[config.LearningMethod](network, rng)


def run_experiment():
    """Run the experiment NumExperiments times, with the seeds InitialSeed, InitialSeed + 1, ..., then summarize."""
    check_settings()
    if int(config.NumExperiments) < 1:
        raise ValueError("NumExperiments must be at least 1")
    folder = os.path.join(config.ResultsDir, config.ExperimentName)
    seeds = [int(config.InitialSeed) + index for index in range(int(config.NumExperiments))]
    results = []
    for number, seed in enumerate(seeds, start=1):
        run = "run %d/%d, seed %d" % (number, len(seeds), seed)
        results.append(run_once(seed, os.path.join(folder, "seed%d" % seed), run))
    print("=" * 100)
    print("Experiment " + config.ExperimentName + ": summary over " + str(len(seeds)) + " run(s)")
    summarize_runs(folder, seeds, results)
    return results


def run_once(seed, folder, run):
    """One run of the experiment with the given seed; its results are written to folder."""
    dtype = np.dtype(config.Dtype).type
    problem_seed, model_seed, learner_seed, calibration_seed = np.random.SeedSequence(int(seed)).spawn(4)
    problem = make_problem(dtype, np.random.default_rng(problem_seed))
    network = Network(problem.spec, dtype, np.random.default_rng(model_seed))
    batch_size = int(config.TrainingBatchSize)
    learner = make_learner(network, np.random.default_rng(learner_seed))
    calibration_rng = np.random.default_rng(calibration_seed)
    observation, context = calibration_contexts(problem, int(config.CalibrationBatchSize), calibration_rng)
    network.calibrate(observation, context)

    names = [problem.task_name(task) for task in range(problem.num_tasks)]
    record = ContinualRecord(names, config.SpeedWindow)
    neurons = sum(layer.num_neurons for layer in network.layers)
    synapses = sum(layer.weight.size for layer in network.layers)
    changed = changed_settings()
    print("=" * 100)
    print("Experiment " + config.ExperimentName + " (" + run + ")")
    print("Settings changed from config.py: " + (", ".join("%s=%r" % item for item in changed.items()) or "none"))
    print("Learning: %s | tasks: %s | neurons: %d | synapses: %d" % (config.LearningMethod, ", ".join(names), neurons,
                                                                    synapses))
    start = time.time()
    for task in range(problem.num_tasks):
        learner.task_started(task)
        label = "[task %d/%d %s]" % (task + 1, problem.num_tasks, names[task])
        curve = train_task(learner, problem.train_environment(task, batch_size), label)
        scores = []
        for evaluated in range(problem.num_tasks):
            environment = problem.test_environment(evaluated, int(config.EvaluationBatchSize))
            scores.append(evaluate(learner, environment))
        record.add_task(task, curve, scores)
        seen = np.mean(scores[: task + 1])
        listing = " ".join("%.3f" % score for score in scores)
        elapsed = time.time() - start
        print("%s test accuracy per task: %s | mean over seen tasks %.3f | %.0fs" % (label, listing, seen, elapsed))
    result = record.save(folder, current_settings(), seed, time.time() - start)
    print("Final mean accuracy %.3f, mean forgetting %.3f; results in %s" % (
        result["final_average_accuracy"], result["mean_forgetting"], folder))
    return result


def calibration_contexts(problem, batch_size, rng):
    """Images of the first task, each paired with the context code of a randomly chosen task.

    Thresholds calibrated this way describe a neuron's average gating state. Under a particular task's context the
    neurons that this context gates more than average fall silent and the others become more active, so every task gets
    its own subset of active neurons. Calibrating with the first task's context only would absorb that task's gating
    into the thresholds, and calibrating without context would leave almost every neuron silent under any context. Only
    the context codes of later tasks are used, never their data.
    """
    observation, context = problem.train_environment(0, batch_size).reset()
    if context is None:
        return observation, None
    codes = np.stack([problem.train_environment(task, 1).reset()[1][0] for task in range(problem.num_tasks)])
    return observation, codes[rng.integers(0, len(codes), batch_size)]


def train_task(learner, environment, label):
    """Run one task's training stream; returns the mean reward of every step."""
    steps = int(config.StepsPerTask)
    log_every = int(config.LogEvery)
    rewards = np.zeros(steps)
    observation, context = environment.reset()
    for step in range(steps):
        actions = learner.act(observation, context)
        observation, context, reward, done, info = environment.step(actions)
        learner.observe(reward, done, info)
        rewards[step] = float(np.mean(reward))
        if log_every > 0 and (step + 1) % log_every == 0:
            stats = learner.stats()
            details = " ".join("%s=%.3g" % (key, value) for key, value in sorted(stats.items()) if key != "reward")
            recent = rewards[max(0, step + 1 - log_every): step + 1].mean()
            print("%s step %d reward %.3f %s" % (label, step + 1, recent, details))
    learner.finish()
    return rewards


def evaluate(learner, environment):
    """Mean episode return of the greedy policy over one pass of the test set (or EvaluationMaxEpisodes episodes)."""
    limit = int(config.EvaluationMaxEpisodes)
    total = environment.episodes_per_pass
    if total is None and limit <= 0:
        raise ValueError("Endless evaluation streams need EvaluationMaxEpisodes > 0")
    if limit > 0:
        total = limit if total is None else min(total, limit)
    observation, context = environment.reset()
    returns = None
    finished = []
    count = 0
    while count < total:
        actions = learner.act_greedy(observation, context)
        observation, context, reward, done, info = environment.step(actions)
        returns = reward.astype(float) if returns is None else returns + reward
        ended = returns[done][: total - count]
        finished.append(ended)
        count += len(ended)
        returns[done] = 0.0
    return float(np.mean(np.concatenate(finished)))


def default_settings():
    """Every setting with the value written in config.py, read afresh from the file (main.py may have changed some).

    Settings are the names in config.py that start with a capital letter.
    """
    namespace = runpy.run_path(ConfigFile)
    return {name: value for name, value in namespace.items() if name[:1].isupper()}


def current_settings():
    """Every setting with the value it has now."""
    return {name: getattr(config, name) for name in default_settings()}


def changed_settings():
    """The settings whose value now differs from the one written in config.py."""
    defaults = default_settings()
    return {name: getattr(config, name) for name in defaults if getattr(config, name) != defaults[name]}


def check_settings():
    """Fail loudly when a name set on config is not a setting of config.py, which is most likely a typo in main.py."""
    defaults = default_settings()
    unknown = sorted(name for name, value in vars(config).items()
                     if not name.startswith("_") and not isinstance(value, types.ModuleType) and name not in defaults)
    if unknown:
        raise ValueError("Unknown setting(s) " + ", ".join(unknown) + "; every setting must be defined in config.py")
