"""Gradient-based learning through the whole network, as a reference for recirculation learning.

The trigger emits its firing probability, so the forward pass is a smooth function of all parameters and the network's
exact backward pass applies. Three objectives are available:

supervised
    Cross-entropy between the softmax of the preferences and the label in info["target"]. The action taken is the
    most preferred one, so the reward reported by the environment is the training accuracy.

action_values
    Reward only, with the readout rule of the recirculation learner: the preferences are predicted rewards, and the
    chosen action's prediction regresses on target = r + gamma * V(next), V being the policy-weighted action values.

actor_critic
    Reward only, one-step actor-critic: the TD error delta = r + gamma * V(next) - V(now) weights the policy gradient
    and the critic regresses on delta.

In the reward-only objectives gradients reach every synapse by backpropagation instead of by the second pass, so the
comparison with the recirculation learner (action_values) isolates the credit-assignment method. All objectives add a
penalty on the mean trigger output of every layer (sparse long-distance activity) and use Adam. Weights are projected
back to [0, weight_max] after each step, because a conductance cannot be negative.
"""

from types import SimpleNamespace

import numpy as np

import config
from learner import Learner, RunningMean, exploring_policy, policy_score, one_hot
from learner import sample_actions, softmax


class Adam:
    """Adam optimiser over a dictionary of NumPy arrays, updated in place."""

    def __init__(self, params, learning_rate, beta1, beta2, epsilon=1e-8):
        self.params = params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first = {name: np.zeros_like(value) for name, value in params.items()}
        self.second = {name: np.zeros_like(value) for name, value in params.items()}
        self.steps = 0


    def step(self, grads):
        self.steps += 1
        correction1 = 1.0 - self.beta1 ** self.steps
        correction2 = 1.0 - self.beta2 ** self.steps
        for name, param in self.params.items():
            grad = grads[name]
            first = self.first[name]
            second = self.second[name]
            first *= self.beta1
            first += (1.0 - self.beta1) * grad
            second *= self.beta2
            second += (1.0 - self.beta2) * grad * grad
            step = self.learning_rate * (first / correction1) / (np.sqrt(second / correction2) + self.epsilon)
            param -= step.astype(param.dtype, copy=False)


class GradientLearner(Learner):
    """Backpropagation with a supervised, an action-value or an actor-critic objective."""

    def __init__(self, network, rng):
        super().__init__(network, rng)
        if config.GradientObjective not in ("supervised", "action_values", "actor_critic"):
            raise ValueError("Unknown gradient objective '" + str(config.GradientObjective) + "'")
        params = network.parameters()
        if not config.LearnThresholds:
            params = {name: value for name, value in params.items() if not name.endswith("threshold")}
        self.optimizer = Adam(params, float(config.GradientLearningRate), float(config.AdamBeta1),
                              float(config.AdamBeta2))
        self.current = None
        self.pending = None
        self.monitor = RunningMean()


    def act(self, observation, context):
        if self.pending is not None:
            self._learn_from_reward(observation, context)
        state = self.network.forward(observation, context)
        if config.GradientObjective == "supervised":
            policy = None
            actions = np.argmax(state.preferences, axis=1)
        else:
            temperature = float(config.ActionTemperature)
            soft, policy = exploring_policy(state.preferences, temperature, float(config.Exploration))
            actions = sample_actions(policy, self.rng)
        self.current = SimpleNamespace(state=state, policy=policy, actions=actions)
        if policy is not None:
            self.current.soft = soft
        for layer, layer_state in zip(self.network.layers, state.layers):
            self.monitor.update(layer.name + ".rate", layer_state.z.mean())
        return actions


    def observe(self, reward, done, info):
        self.monitor.update("reward", np.mean(reward))
        if config.GradientObjective == "supervised":
            if "target" not in info:
                raise ValueError("The supervised objective needs info['target'] from the environment")
            self._learn_supervised(np.asarray(info["target"]))
        else:
            self.pending = (np.asarray(reward, dtype=self.network.dtype), np.asarray(done, dtype=bool))


    def finish(self):
        if self.pending is not None:
            self._learn_from_reward(None, None)


    def _sparsity_grads(self, state):
        weight = float(config.SparsityWeight)
        return [np.full(layer_state.z.shape, weight / layer_state.z.size, dtype=self.network.dtype)
                for layer_state in state.layers]


    def _apply(self, state, grad_preferences, grad_value):
        grads = self.network.backward(state, grad_preferences, grad_value, self._sparsity_grads(state))
        self.optimizer.step(grads)
        self.network.constrain()


    def _learn_supervised(self, target):
        state = self.current.state
        batch = len(target)
        probabilities = softmax(state.preferences, 1.0)
        labels = one_hot(target, self.network.num_actions, self.network.dtype)
        loss = -np.mean(np.log(np.maximum(np.sum(probabilities * labels, axis=1), 1e-12)))
        self.monitor.update("loss", loss)
        self._apply(state, (probabilities - labels) / batch, np.zeros(batch, dtype=self.network.dtype))


    def _state_value(self, state, policy):
        if config.GradientObjective == "action_values":
            return np.sum(policy * state.preferences, axis=1)
        return state.value


    def _learn_from_reward(self, next_observation, next_context):
        reward, done = self.pending
        self.pending = None
        state = self.current.state
        policy = self.current.policy
        batch = len(reward)
        temperature = float(config.ActionTemperature)
        continuing = (~done).astype(self.network.dtype)
        next_value = 0.0
        if next_observation is not None and continuing.any():
            following = self.network.forward(next_observation, next_context)
            _, next_policy = exploring_policy(following.preferences, temperature, float(config.Exploration))
            next_value = self._state_value(following, next_policy)
        target = reward + float(config.Gamma) * continuing * next_value
        delta = target - self._state_value(state, policy)
        self.monitor.update("abs_td_error", np.abs(delta).mean())
        if config.GradientObjective == "action_values":
            # Loss 0.5 * mean((target - Q(s, a))^2) on the chosen action only; the target is held fixed.
            rows = np.arange(batch)
            grad_preferences = np.zeros_like(state.preferences)
            grad_preferences[rows, self.current.actions] = -(target - state.preferences[rows, self.current.actions]) / batch
            self._apply(state, grad_preferences, np.zeros(batch, dtype=self.network.dtype))
            return
        soft = self.current.soft
        score = policy_score(self.current.actions, soft, temperature)
        # Policy loss -mean(delta * log pi(action)). Entropy bonus -c * mean(H) on the softmax part, whose derivative
        # with respect to the preferences is -soft_j (log soft_j + H) / temperature.
        log_soft = np.log(np.maximum(soft, 1e-12))
        entropy = -np.sum(soft * log_soft, axis=1)
        grad_entropy = float(config.EntropyWeight) * soft * (log_soft + entropy[:, None]) / temperature
        grad_preferences = (-delta[:, None] * score + grad_entropy) / batch
        grad_value = -float(config.CriticLossWeight) * delta / batch
        self._apply(state, grad_preferences, grad_value)


    def stats(self):
        return self.monitor.as_dict()
