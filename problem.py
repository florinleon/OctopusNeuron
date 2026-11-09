"""What a problem has to provide so that any learner and network can be trained on it.

A problem is a sequence of tasks. For every task it can create a training environment and a test environment. An
environment is batched: it runs batch_size independent streams side by side, and every call to step() advances all of
them by one step. Episodes may end at different times in different streams; done marks the streams whose episode just
ended, and their next observation already belongs to a new episode.

The network only sees the ProblemSpec: observation size, observation positions in the unit square (None when the
observation has no spatial layout), the size of the context code (0 when there is none) and the number of actions.
"""


class ProblemSpec:
    """Static description of observations, context and actions shared by all tasks of a problem."""

    def __init__(self, observation_size, observation_positions, context_size, num_actions):
        self.observation_size = observation_size
        self.observation_positions = observation_positions
        self.context_size = context_size
        self.num_actions = num_actions


class Environment:
    """Batched environment interface.

    reset() -> (observation (batch, size), context (batch, context_size) or None)
    step(actions) -> (observation, context, reward (batch,), done (batch,) bool, info dict)

    episodes_per_pass is the number of episodes in one complete pass over a finite test set (None for endless
    streams); evaluation uses it to measure every test item exactly once.
    """

    episodes_per_pass = None


    def reset(self):
        raise NotImplementedError


    def step(self, actions):
        raise NotImplementedError


class Problem:
    """A sequence of tasks with a common ProblemSpec."""

    spec = None
    num_tasks = 0


    def task_name(self, task):
        return "task " + str(task)


    def train_environment(self, task, batch_size):
        raise NotImplementedError


    def test_environment(self, task, batch_size):
        raise NotImplementedError
