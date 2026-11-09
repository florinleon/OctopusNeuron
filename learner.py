"""Interface shared by all learning methods, and small policy helpers.

A learner drives a network through a batched stream of environment steps:

    actions = learner.act(observation, context)     choose actions (and learn from the previous step if pending)
    learner.observe(reward, done, info)             receive the outcome of those actions
    learner.finish()                                apply any update still waiting for the next observation

Updates may be deferred until the next act(): temporal-difference errors need the value of the next state, which only
becomes known when the next observation arrives. act_greedy() is used for evaluation and never changes the learner.
"""

import numpy as np


class Learner:
    """Base class; subclasses implement act, observe and optionally finish, task_started and stats."""

    def __init__(self, network, rng):
        self.network = network
        self.rng = rng


    def act(self, observation, context):
        raise NotImplementedError


    def observe(self, reward, done, info):
        raise NotImplementedError


    def finish(self):
        """Apply pending updates at the end of a training phase (the last step is treated as terminal)."""


    def task_started(self, task):
        """Called by the harness when the training stream switches to a new task; most learners ignore it."""


    def act_greedy(self, observation, context):
        """Action with the highest preference, without exploration and without learning."""
        state = self.network.forward(observation, context)
        return np.argmax(state.preferences, axis=1)


    def stats(self):
        """Running diagnostics for logging, as a flat dictionary of floats."""
        return {}


def softmax(preferences, temperature):
    scaled = preferences / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=1, keepdims=True)


def exploring_policy(preferences, temperature, exploration):
    """Softmax policy mixed with a uniform floor: (1 - exploration) * softmax + exploration / num_actions.

    The floor keeps every action tried now and then. Without it, a policy that became confident on the classes of
    earlier tasks would almost never try the classes of a new task, and reward-only learning would get no signal.
    Returns the softmax part and the mixed policy actually sampled from.
    """
    soft = softmax(preferences, temperature)
    mixed = (1.0 - exploration) * soft + exploration / preferences.shape[1]
    return soft, mixed


def policy_score(actions, soft, temperature):
    """Softmax score (onehot(action) - soft) / temperature, one row per stream.

    Actions are sampled from the mixed policy, but they are credited as if the softmax part had chosen them. The exact
    score of the mixture is scaled by soft(action) / mixed(action), which is close to zero precisely for the useful
    case: an action the softmax had all but ruled out (a class of a new task) that the uniform floor tried and that
    was rewarded. Crediting it in full lets one lucky try move the policy at once; the price is a small bias towards
    the actions the floor happens to try, which vanishes as exploration goes to zero.
    """
    return (one_hot(actions, soft.shape[1], soft.dtype) - soft) / temperature


def sample_actions(policy, rng):
    """Draw one action per row of a batch of categorical distributions."""
    thresholds = rng.random((policy.shape[0], 1))
    actions = np.sum(np.cumsum(policy, axis=1) < thresholds, axis=1)
    return np.minimum(actions, policy.shape[1] - 1)


def one_hot(actions, num_actions, dtype):
    encoded = np.zeros((len(actions), num_actions), dtype=dtype)
    encoded[np.arange(len(actions)), actions] = 1.0
    return encoded


class RunningMean:
    """Exponential moving averages of named scalars, for compact progress logs."""

    def __init__(self, smoothing=0.98):
        self.smoothing = smoothing
        self.values = {}


    def update(self, name, value):
        value = float(value)
        if name in self.values:
            self.values[name] = self.smoothing * self.values[name] + (1.0 - self.smoothing) * value
        else:
            self.values[name] = value


    def as_dict(self):
        return dict(self.values)
