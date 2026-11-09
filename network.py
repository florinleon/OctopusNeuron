"""A network of unipolar neuron layers with a linear readout of the long-distance outputs.

The network knows nothing about the problem it serves except what a problem specification states: the size of an
observation, the positions of observation units (or None), the size of the context code and the number of actions.

Layers are built in the order of config.Layers. Each layer samples its inputs from named sources:

    observation   the observation vector of the problem
    <layer>.y     local outputs of an earlier layer, positioned at their neurons
    <layer>.z     trigger (long-distance) outputs of an earlier layer

Local wiring takes sources near the neuron; global wiring may take any source. The readout is linear in the
concatenation of the sources listed in config.ReadoutSources, normally trigger outputs only. It produces one preference
per action (action values or policy logits, depending on the learner) and one state value (used by critic-based rules).
"""

from types import SimpleNamespace

import numpy as np

import config
from neurons import NeuronLayer
from morphology import Morphology
from wiring import grid_positions, sample_sources


ObservationSource = "observation"


class Network:
    """Layers of neurons plus readout, with forward, backward, recirculation, calibration and parameter access."""

    def __init__(self, spec, dtype, rng):
        self.dtype = dtype
        self.morphology = Morphology()
        self.context_size = int(spec.context_size)
        self.num_actions = int(spec.num_actions)
        positions = spec.observation_positions
        if positions is None:
            positions = np.full((spec.observation_size, 2), 0.5)
        self.source_sizes = {ObservationSource: int(spec.observation_size)}
        self.source_positions = {ObservationSource: np.asarray(positions, dtype=np.float64)}
        self.layers = []
        self.layer_blocks = []
        for layer_config in config.Layers:
            self._add_layer(layer_config, rng)
        self.readout_sources = list(config.ReadoutSources)
        self.readout_bias = bool(config.ReadoutBias)
        for name in self.readout_sources:
            if name not in self.source_sizes:
                raise ValueError("Readout source '" + name + "' does not exist")
        self.num_features = sum(self.source_sizes[name] for name in self.readout_sources)
        self.actor_weight = np.zeros((self.num_actions, self.num_features), dtype=dtype)
        self.actor_bias = np.zeros(self.num_actions, dtype=dtype)
        self.critic_weight = np.zeros(self.num_features, dtype=dtype)
        self.critic_bias = np.zeros(1, dtype=dtype)


    def _add_layer(self, layer_config, rng):
        positions = grid_positions(int(layer_config["grid"][0]), int(layer_config["grid"][1]),
                                   int(layer_config["neurons_per_position"]))
        index_parts = []
        blocks = []
        offset = 0
        for spec in layer_config["inputs"]:
            if spec["source"] not in self.source_sizes:
                raise ValueError("Layer '" + layer_config["name"] + "' reads unknown source '" + spec["source"] + "'")
            index_parts.append(sample_sources(positions, self.source_positions[spec["source"]], spec, rng) + offset)
            blocks.append((spec["source"], self.source_sizes[spec["source"]]))
            offset += self.source_sizes[spec["source"]]
        wiring = SimpleNamespace(positions=positions, source_index=np.concatenate(index_parts, axis=1), input_size=offset)
        layer = NeuronLayer(layer_config["name"], self.morphology, wiring, self.context_size, self.dtype, rng)
        self.layers.append(layer)
        self.layer_blocks.append(blocks)
        # Local outputs are stored neuron by neuron, so output j of neuron c is entry c * num_outputs + j.
        self.source_sizes[layer_config["name"] + ".y"] = layer.num_neurons * layer.num_outputs
        self.source_positions[layer_config["name"] + ".y"] = np.repeat(positions, layer.num_outputs, axis=0)
        self.source_sizes[layer_config["name"] + ".z"] = layer.num_neurons
        self.source_positions[layer_config["name"] + ".z"] = positions


    def parameters(self):
        """All learned arrays by qualified name (layer.parameter or readout.parameter); biases only if enabled."""
        params = {}
        for layer in self.layers:
            for key, value in layer.parameters().items():
                params[layer.name + "." + key] = value
        params["readout.actor_weight"] = self.actor_weight
        params["readout.critic_weight"] = self.critic_weight
        if self.readout_bias:
            params["readout.actor_bias"] = self.actor_bias
            params["readout.critic_bias"] = self.critic_bias
        return params


    def constrain(self):
        for layer in self.layers:
            layer.constrain()


    def _layer_inputs(self, values, blocks):
        if len(blocks) == 1:
            return values[blocks[0][0]]
        return np.concatenate([values[name] for name, _ in blocks], axis=1)


    def forward(self, observation, context):
        """Run all layers and the readout; returns a state with per-layer states, features, preferences and value."""
        batch = observation.shape[0]
        values = {ObservationSource: observation}
        layer_states = []
        for layer, blocks in zip(self.layers, self.layer_blocks):
            state = layer.forward(self._layer_inputs(values, blocks), context)
            values[layer.name + ".y"] = state.y.reshape(batch, -1)
            values[layer.name + ".z"] = state.z
            layer_states.append(state)
        features = np.concatenate([values[name] for name in self.readout_sources], axis=1)
        preferences = features @ self.actor_weight.T + self.actor_bias
        value = features @ self.critic_weight + self.critic_bias[0]
        return SimpleNamespace(layers=layer_states, values=values, features=features, preferences=preferences, value=value)


    def backward(self, state, grad_preferences, grad_value, grad_trigger=None):
        """Gradients of a loss given its derivatives with respect to the preferences (batch, actions) and value (batch).

        grad_trigger optionally adds direct derivatives with respect to each layer's trigger outputs (a list with one
        array (batch, neurons) or None per layer), which is how penalties on long-distance activity enter.
        Returns a dictionary with the same keys as parameters().
        """
        batch = state.features.shape[0]
        grads = {
            "readout.actor_weight": grad_preferences.T @ state.features,
            "readout.actor_bias": grad_preferences.sum(axis=0),
            "readout.critic_weight": state.features.T @ grad_value,
            "readout.critic_bias": np.array([grad_value.sum()], dtype=self.dtype),
        }
        grad_features = grad_preferences @ self.actor_weight + grad_value[:, None] * self.critic_weight[None, :]
        source_grads = {}
        offset = 0
        for name in self.readout_sources:
            size = self.source_sizes[name]
            self._accumulate(source_grads, name, grad_features[:, offset:offset + size])
            offset += size
        for index in reversed(range(len(self.layers))):
            layer = self.layers[index]
            grad_y = source_grads.get(layer.name + ".y", np.zeros((batch, layer.num_neurons * layer.num_outputs), self.dtype))
            grad_z = source_grads.get(layer.name + ".z", np.zeros((batch, layer.num_neurons), self.dtype))
            if grad_trigger is not None and grad_trigger[index] is not None:
                grad_z = grad_z + grad_trigger[index]
            shaped_grad_y = grad_y.reshape(batch, layer.num_neurons, layer.num_outputs)
            layer_grads, grad_inputs = layer.backward(state.layers[index], shaped_grad_y, grad_z)
            for key, value in layer_grads.items():
                grads[layer.name + "." + key] = value
            offset = 0
            for name, size in self.layer_blocks[index]:
                if name != ObservationSource:
                    self._accumulate(source_grads, name, grad_inputs[:, offset:offset + size])
                offset += size
        return grads


    def recirculate(self, state, feature_change):
        """Top-down second pass of recirculation learning; returns one result of NeuronLayer.recirculate per layer.

        feature_change (batch, num_features) holds the change that the readout asks of each of its sources. Layers are
        visited from the last to the first: each layer turns the changes asked of its outputs into shifts of its
        segments, and its contacts pass changes back to the outputs of the earlier layers they listen to. What travels
        between layers is a requested change of activity, routed backwards through the same contacts, and every layer
        handles it in the same way; no layer computes anything for another one.
        """
        batch = state.features.shape[0]
        requested = {}
        offset = 0
        for name in self.readout_sources:
            size = self.source_sizes[name]
            self._accumulate(requested, name, feature_change[:, offset:offset + size])
            offset += size
        results = [None] * len(self.layers)
        for index in reversed(range(len(self.layers))):
            layer = self.layers[index]
            output_change = requested.get(layer.name + ".y")
            if output_change is not None:
                output_change = output_change.reshape(batch, layer.num_neurons, layer.num_outputs)
            trigger_change = requested.get(layer.name + ".z", np.zeros((batch, layer.num_neurons), self.dtype))
            # Observations do not learn, so a layer that reads only observations sends nothing back.
            send_back = any(name != ObservationSource for name, _ in self.layer_blocks[index])
            results[index] = layer.recirculate(state.layers[index], output_change, trigger_change, send_back)
            offset = 0
            for name, size in self.layer_blocks[index]:
                if name != ObservationSource:
                    self._accumulate(requested, name, results[index].returned[:, offset:offset + size])
                offset += size
        return results


    def _accumulate(self, source_grads, name, grad):
        if name in source_grads:
            source_grads[name] = source_grads[name] + grad
        else:
            source_grads[name] = grad


    def calibrate(self, observation, context):
        """Calibrate layer after layer on one batch, feeding each layer the outputs of the already calibrated ones."""
        batch = observation.shape[0]
        values = {ObservationSource: observation}
        for layer, blocks in zip(self.layers, self.layer_blocks):
            inputs = self._layer_inputs(values, blocks)
            layer.calibrate(inputs, context)
            state = layer.forward(inputs, context)
            values[layer.name + ".y"] = state.y.reshape(batch, -1)
            values[layer.name + ".z"] = state.z
