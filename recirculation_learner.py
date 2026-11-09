"""Recirculation learning: every synapse learns from how its own segment moves between two passes through the network.

First pass
    The observation passes through the layers to the readout, which predicts the reward of every action. An action is
    chosen from a softmax over the predictions mixed with a small uniform share, and the environment returns a reward.

Second pass
    The readout's prediction for the chosen action is pulled towards the target r + gamma * V(next). The readout sends
    this back through its contacts: each of its sources is asked to change by error * u[a, source], where u[a] holds the
    readout weights of the chosen action and error = target - Q(s, a); the return strength of a contact equals its
    forward weight. The request enters the neurons of the last layer at their triggers, the neurons settle again, and
    their contacts pass requests on to the layers they listen to, whose neurons respond in the same way
    (Network.recirculate and NeuronLayer.recirculate describe the physics). The pass is computed in the limit of a small
    pull, where the response is exact and linear; the size of the pull then only scales every change and is absorbed
    into the learning rate.

Learning
    Every synapse compares its own segment in the two passes and changes by
        dw_i = SynapseRate * x_i * (E_i - v_k) * shift_k / (1 + ConsolidationStrength * importance_i),
    presynaptic activity times driving force times how far its segment moved between the passes. The rule and the
    signals it uses are the same in every layer, so adding layers needs nothing new. Because return strengths equal
    forward weights, these changes follow the gradient of the squared readout error, except for the way conductances
    shape the spread of the back-boost, which the second pass leaves out.

    The readout learns by a normalised least-mean-squares step: only the chosen action's prediction moves towards the
    target, so a confident wrong action is corrected at once and the classes of a new task can enter a policy that
    earlier tasks have made confident.

    Trigger outputs are firing probabilities, so the forward pass is deterministic and exploration comes only from the
    choice of action. Thresholds keep their calibrated values: learning them would act like learning biases, which
    shift with every task and move the representations that earlier tasks rely on.

Continual learning
    Consolidation slows synapses that changed a lot recently: importance is a slowly decaying sum of |dw|. Context gating
    protects in part without any extra rule: a branch shunted at its entry hardly moves when current enters the trigger,
    so its synapses change little while another task is learned, although requests returned to its own local outputs
    still reach it.
"""

from types import SimpleNamespace

import numpy as np

import config
from learner import Learner, RunningMean, exploring_policy, one_hot, sample_actions


class RecirculationLearner(Learner):
    """Online learner with two passes per step: the forward pass, then a top-down pass started by the readout error."""

    def __init__(self, network, rng):
        super().__init__(network, rng)
        self.gamma = float(config.Gamma)
        self.importance = [np.zeros(layer.weight.shape, network.dtype) for layer in network.layers]
        self.previous = None
        self.pending = None
        self.monitor = RunningMean()


    def act(self, observation, context):
        # The pending update is applied first, so the new observation already meets the updated weights.
        if self.pending is not None:
            self._learn(observation, context)
        state = self.network.forward(observation, context)
        _, policy = self._policy(state)
        actions = sample_actions(policy, self.rng)
        self.previous = SimpleNamespace(state=state, actions=actions)
        for layer, layer_state in zip(self.network.layers, state.layers):
            self.monitor.update(layer.name + ".rate", layer_state.z.mean())
        return actions


    def observe(self, reward, done, info):
        self.pending = (np.asarray(reward, dtype=self.network.dtype), np.asarray(done, dtype=bool))
        self.monitor.update("reward", np.mean(reward))


    def finish(self):
        if self.pending is not None:
            self._learn(None, None)


    def _policy(self, state):
        return exploring_policy(state.preferences, float(config.ActionTemperature), float(config.Exploration))


    def synaptic_steps(self, state, actions, error):
        """Second pass for a forward state; returns x_i * (E_i - v_k) * shift_k averaged over the batch, per layer.

        actions (batch,) are the chosen actions and error (batch,) the pull on their predictions. The result is one
        array (neurons, synapses) per layer, together with the per-layer results of the second pass for diagnostics.
        """
        feature_change = error[:, None] * self.network.actor_weight[actions]
        results = self.network.recirculate(state, feature_change)
        batch = len(error)
        steps = [np.einsum("nbs,nbs->ns", result.factor, layer_state.x) / batch
                 for layer_state, result in zip(state.layers, results)]
        return steps, results


    def _learn(self, next_observation, next_context):
        reward, done = self.pending
        self.pending = None
        state = self.previous.state
        actions = self.previous.actions
        batch = len(reward)
        continuing = (~done).astype(self.network.dtype)
        # The value of the next state matters only for streams whose episode goes on; one-step episodes skip it.
        next_value = 0.0
        if next_observation is not None and continuing.any():
            following = self.network.forward(next_observation, next_context)
            _, next_policy = self._policy(following)
            next_value = np.sum(next_policy * following.preferences, axis=1)
        target = reward + self.gamma * continuing * next_value
        error = target - state.preferences[np.arange(batch), actions]

        # The second pass uses the readout weights that made the prediction, so it is run before the readout learns.
        steps, results = self.synaptic_steps(state, actions, error)
        for index, layer in enumerate(self.network.layers):
            plasticity = 1.0 / (1.0 + float(config.ConsolidationStrength) * self.importance[index])
            change = plasticity * float(config.SynapseRate) * steps[index]
            layer.weight += change
            layer.constrain()
            self.importance[index] *= 1.0 - float(config.ConsolidationDecay)
            self.importance[index] += np.abs(change)
            self.monitor.update(layer.name + ".shift", np.abs(results[index].shift).mean())
            self.monitor.update(layer.name + ".change", np.abs(change).mean())
            self.monitor.update(layer.name + ".plasticity", plasticity.mean())

        # Readout: normalised least-mean-squares step on the chosen action's prediction only.
        network = self.network
        scale = float(config.ValueRate) / (batch * (1.0 + float(np.mean(np.sum(state.features ** 2, axis=1)))))
        pulled = one_hot(actions, network.num_actions, network.dtype) * error[:, None]
        network.actor_weight += scale * (pulled.T @ state.features)
        if network.readout_bias:
            network.actor_bias += scale * pulled.sum(axis=0)
        self.monitor.update("abs_readout_error", np.abs(error).mean())


    def stats(self):
        return self.monitor.as_dict()
