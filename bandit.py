"""A labelled dataset turned into a contextual bandit.

Every episode is a single step: the environment shows one item, the agent picks one of the classes as its action and
receives reward 1 for the correct class and 0 otherwise. The label is also returned in info["target"], so the same
environment serves reward-only learners and supervised reference learners. Training environments draw items in a fresh
random order on every pass; test environments go through the items in a fixed order, so that one pass measures each
test item exactly once.
"""

import numpy as np

from problem import Environment


class DatasetBandit(Environment):
    """One-step episodes over (items, labels), optionally with a fixed feature permutation and a fixed context code."""

    def __init__(self, items, labels, batch_size, shuffle, rng, dtype, scale=1.0, permutation=None, context=None):
        self.items = items
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng
        self.dtype = dtype
        self.scale = scale
        self.permutation = permutation
        self.context = None
        if context is not None:
            self.context = np.tile(np.asarray(context, dtype=dtype), (batch_size, 1))
        self.episodes_per_pass = len(labels)
        self.order = np.arange(len(labels))
        self.position = 0
        self.current_labels = None


    def _new_pass(self):
        self.order = self.rng.permutation(len(self.labels)) if self.shuffle else np.arange(len(self.labels))
        self.position = 0


    def _next_batch(self):
        chosen = []
        needed = self.batch_size
        while needed > 0:
            if self.position >= len(self.order):
                self._new_pass()
            taken = self.order[self.position:self.position + needed]
            chosen.append(taken)
            self.position += len(taken)
            needed -= len(taken)
        index = np.concatenate(chosen)
        self.current_labels = self.labels[index]
        observation = self.items[index].astype(self.dtype) * self.dtype(self.scale)
        if self.permutation is not None:
            observation = observation[:, self.permutation]
        return observation, self.context


    def reset(self):
        self._new_pass()
        return self._next_batch()


    def step(self, actions):
        target = self.current_labels
        reward = (np.asarray(actions) == target).astype(self.dtype)
        done = np.ones(self.batch_size, dtype=bool)
        observation, context = self._next_batch()
        return observation, context, reward, done, {"target": target}
