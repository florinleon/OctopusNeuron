"""A population of unipolar neurons with identical shape: exact forward and backward passes, and the recirculation pass.

For one neuron and one input, the forward pass is:

1. Every synapse i (on segment k(i)) opens a channel of conductance g_i = w_i * x_i, where x_i >= 0 is the activity of
   its source and w_i >= 0 its weight. Excitatory synapses pull their segment towards E_exc, inhibitory ones towards
   E_inh. Context units add fixed inhibitory conductance on gate segments.
2. The neuron settles: M v = drive, with M = diag(leak) + Laplacian + diag(sum of conductances on each segment) and
   drive_k = sum of g_i * E_i over the synapses on segment k. Each segment value is therefore a weighted average of
   its neighbours, of the reversal values of its open channels, and of zero (the leak).
3. The same matrix gives r = M^-1 e_trigger: how one unit of current injected at the trigger spreads into the tree.
   Because M is symmetric, r_k is also how strongly a unit of current at segment k moves the trigger.
4. The trigger computes the firing probability p = sigmoid((v_trigger - threshold) / T); its output z is p itself,
   a real number between 0 and 1.
5. Back-boost: a current back_boost * z, proportional to the output, enters the trigger and spreads by the same rule, so
   every segment rises by back_boost * z * r_k. Branches silenced by inhibition receive less of it.
6. Local outputs read their own segment: y_k = max(0, boosted_k - output_threshold_k).

Array layouts. Per-segment and per-neuron arrays are (batch, neurons, segments) and (batch, neurons). Per-synapse
arrays, by far the largest, are neuron-major: (neurons, batch, synapses). With that layout every transfer between
segments and synapses is one contiguous batched matrix product per neuron (with a fixed one-hot placement), and the
elementwise work of the learning rules runs over contiguous memory.
"""

from types import SimpleNamespace

import numpy as np

import config


class NeuronLayer:
    """Neurons of one layer: their synapses, parameters, forward pass, backward pass and calibration."""

    def __init__(self, name, morphology, wiring, context_size, dtype, rng):
        """Build the synapses of every neuron.

        wiring.positions (neurons, 2) places the neurons in the unit square, wiring.input_size is the length of the
        layer's input vector and wiring.source_index (neurons, fan_in) holds, for every neuron, the indices of its
        sources in it. Each source receives config.ExcitatoryPerSource excitatory and config.InhibitoryPerSource
        inhibitory synapses, each placed on an independently drawn segment. Placing the excitatory and the inhibitory
        contact of a source on different segments is what lets a single source help one part of the tree while
        suppressing another.
        """
        self.name = name
        self.morphology = morphology
        self.dtype = dtype
        self.positions = wiring.positions
        self.input_size = int(wiring.input_size)
        self.num_neurons = len(wiring.positions)
        self.num_segments = morphology.num_segments
        self.num_outputs = morphology.num_outputs
        self.context_size = context_size
        self.back_boost = float(config.BackBoost)
        self.weight_max = float(config.WeightMax)
        self.temperature_factor = float(config.FiringTemperature)
        self.reversal_excitatory = float(config.ReversalExcitatory)
        self.reversal_inhibitory = float(config.ReversalInhibitory)

        excitatory = int(config.ExcitatoryPerSource)
        inhibitory = int(config.InhibitoryPerSource)
        source_index = wiring.source_index
        self.syn_source = np.concatenate([source_index] * (excitatory + inhibitory), axis=1)
        self.num_synapses = self.syn_source.shape[1]
        reversal = np.full(self.num_synapses, self.reversal_inhibitory)
        reversal[: excitatory * source_index.shape[1]] = self.reversal_excitatory
        self.syn_reversal = np.broadcast_to(reversal, self.syn_source.shape).astype(dtype)
        if config.SynapsePlacement == "any":
            allowed = np.arange(self.num_segments)
        elif config.SynapsePlacement == "branches":
            allowed = np.flatnonzero(np.arange(self.num_segments) != morphology.trigger)
        else:
            raise ValueError("Unknown synapse placement '" + str(config.SynapsePlacement) + "'")
        self.syn_segment = rng.choice(allowed, size=self.syn_source.shape)
        self.syn_output = morphology.output_index[self.syn_segment]

        # One-hot placement (neurons, synapses, segments). Moving values between synapses and segments is then a
        # batched matrix product per neuron, far faster than index-based scattering at these sizes.
        placement = np.zeros(self.syn_source.shape + (self.num_segments,), dtype=dtype)
        placement[np.arange(self.num_neurons)[:, None], np.arange(self.num_synapses)[None, :], self.syn_segment] = 1.0
        self._to_segments = placement
        # Plain placement side by side with the reversal-weighted one: one product yields both the conductance per
        # segment (matrix diagonal) and the drive per segment (right-hand side); transposed, it maps a pair of
        # per-segment tables (offset, slope) to offset_k + E_i * slope_k at every synapse.
        paired = np.concatenate([placement, placement * self.syn_reversal[..., None]], axis=2)
        self._to_segments_paired = paired
        self._to_synapses_paired = np.ascontiguousarray(paired.transpose(0, 2, 1))
        # Reverse map from synapses to layer inputs for gradients: synapse values ordered by source, summed per run.
        flat_source = self.syn_source.ravel()
        self._source_order = np.argsort(flat_source, kind="stable")
        sorted_source = flat_source[self._source_order]
        self._source_starts = np.flatnonzero(np.r_[True, sorted_source[1:] != sorted_source[:-1]])
        self._source_present = sorted_source[self._source_starts]

        # Learned parameters. Weights start uniform around one and are rescaled by calibrate().
        self.weight = rng.uniform(0.5, 1.5, size=self.syn_source.shape).astype(dtype)
        self.threshold = np.zeros(self.num_neurons, dtype=dtype)
        self.output_threshold = np.zeros((self.num_neurons, self.num_outputs), dtype=dtype)
        self.temperature = 1.0

        # Fixed context gating: context unit q inhibits gate segment j of neuron c with probability ContextGateFraction,
        # and the trigger segment with probability ContextTrunkGateFraction. Inhibiting a branch entry isolates that
        # branch; inhibiting the trunk vetoes the neuron's long-distance output while its branches keep working locally.
        gates = morphology.gate_segments
        strength = float(config.ContextStrength)
        self.context_conductance = np.zeros((self.num_neurons, context_size, self.num_segments), dtype=dtype)
        if context_size > 0 and len(gates) > 0:
            mask = rng.random((self.num_neurons, context_size, len(gates))) < float(config.ContextGateFraction)
            self.context_conductance[:, :, gates] = mask * strength
        if context_size > 0:
            trunk = rng.random((self.num_neurons, context_size)) < float(config.ContextTrunkGateFraction)
            self.context_conductance[:, :, morphology.trigger] = trunk * strength

        self._gather_cache = {}
        self.base_diagonal = morphology.base_diagonal.astype(dtype)
        self.unit_trigger = np.zeros(self.num_segments, dtype=dtype)
        self.unit_trigger[morphology.trigger] = 1.0


    def parameters(self):
        """Learned arrays by name; learners update them in place so these references stay valid."""
        return {"weight": self.weight, "threshold": self.threshold, "output_threshold": self.output_threshold}


    def constrain(self):
        """Keep weights inside [0, weight_max]; a conductance cannot be negative."""
        np.clip(self.weight, 0.0, self.weight_max, out=self.weight)


    def gather_inputs(self, inputs):
        """Activity of every synapse's source: (batch, input_size) -> (neurons, batch, synapses).

        A flat take with an index precomputed per batch size is several times faster than broadcast fancy indexing.
        """
        batch = inputs.shape[0]
        if batch not in self._gather_cache:
            offsets = np.arange(batch)[None, :, None] * self.input_size
            self._gather_cache[batch] = self.syn_source[:, None, :] + offsets
        return np.take(np.ascontiguousarray(inputs).reshape(-1), self._gather_cache[batch])


    def segment_sum(self, values):
        """Sum per-synapse values (neurons, batch, synapses) into their segments (batch, neurons, segments)."""
        return np.matmul(values, self._to_segments).transpose(1, 0, 2)


    def at_synapses_affine(self, offset, slope):
        """Per-synapse value offset_k + E_i * slope_k, with k the synapse's segment and E_i its reversal value.

        Learning rules need factors such as (E_i - v_k) * r_k at every synapse. They depend on the synapse only through
        its segment and its type, so they are formed per segment (batch, neurons, segments) and moved to the synapses by
        a single product; the result is (neurons, batch, synapses).
        """
        table = np.concatenate([offset, slope], axis=2)
        return np.matmul(table.transpose(1, 0, 2), self._to_synapses_paired)


    def outputs_to_segments(self, values):
        """Place per-output values (batch, neurons, outputs) on their segments; other segments get zero."""
        placed = np.zeros(values.shape[:2] + (self.num_segments,), dtype=self.dtype)
        placed[..., self.morphology.output_segments] = values
        return placed


    def input_sum(self, values):
        """Sum per-synapse values (neurons, batch, synapses) back onto the layer inputs (batch, input_size)."""
        batch = values.shape[1]
        ordered = values.transpose(1, 0, 2).reshape(batch, -1)[:, self._source_order]
        result = np.zeros((batch, self.input_size), dtype=self.dtype)
        result[:, self._source_present] = np.add.reduceat(ordered, self._source_starts, axis=1)
        return result


    def settle(self, inputs, context):
        """Steady state of every neuron: returns synaptic input activity, matrix diagonal, values v and spread r."""
        x = self.gather_inputs(inputs)
        sums = np.matmul(self.weight[:, None, :] * x, self._to_segments_paired).transpose(1, 0, 2)
        diagonal = self.base_diagonal + sums[..., : self.num_segments]
        drive = sums[..., self.num_segments:]
        if self.context_size > 0 and context is not None:
            gating = np.tensordot(context, self.context_conductance, axes=([1], [1]))
            diagonal = diagonal + gating
            drive = drive + gating * self.reversal_inhibitory
        rhs = np.stack([drive, np.broadcast_to(self.unit_trigger, drive.shape)], axis=-1)
        solution = self.morphology.solver.solve(diagonal, rhs)
        return x, diagonal, solution[..., 0], solution[..., 1]


    def forward(self, inputs, context):
        """Full forward pass of the layer for a batch of inputs (batch, input_size) and context (batch, Q) or None.

        z is the firing probability, which makes the pass deterministic and differentiable.
        """
        x, diagonal, v, r = self.settle(inputs, context)
        trigger = self.morphology.trigger
        outputs = self.morphology.output_segments
        activation = (v[..., trigger] - self.threshold) / self.temperature
        p = (0.5 * (1.0 + np.tanh(0.5 * activation))).astype(self.dtype, copy=False)
        z = p
        boosted_out = v[..., outputs] + self.back_boost * z[..., None] * r[..., outputs]
        y = np.maximum(boosted_out - self.output_threshold, 0.0)
        return SimpleNamespace(x=x, diagonal=diagonal, v=v, r=r, p=p, z=z, boosted_out=boosted_out, y=y)


    def backward(self, state, grad_y, grad_z):
        """Exact gradients for a deterministic forward state, given dLoss/dy (batch, neurons, outputs) and dLoss/dz.

        Returns the parameter gradients and dLoss/dinputs (batch, input_size). The two adjoint systems (for v and for
        the spread r used by the back-boost) share the matrix of the forward pass, so they are solved together.
        """
        trigger = self.morphology.trigger
        outputs = self.morphology.output_segments
        grad_out = grad_y * (state.boosted_out > self.output_threshold)
        grad_output_threshold = -grad_out.sum(axis=0)

        grad_boosted = np.zeros_like(state.v)
        grad_boosted[..., outputs] = grad_out
        grad_z_total = grad_z + self.back_boost * np.sum(grad_boosted * state.r, axis=-1)
        grad_r = self.back_boost * state.z[..., None] * grad_boosted
        grad_activation = grad_z_total * state.p * (1.0 - state.p) / self.temperature
        # The trigger carries no local output, so its entry of grad_boosted is zero and can take the trigger gradient.
        grad_v = grad_boosted
        grad_v[..., trigger] += grad_activation
        grad_threshold = -grad_activation.sum(axis=0)

        adjoint = self.morphology.solver.solve(state.diagonal, np.stack([grad_v, grad_r], axis=-1))
        lam = adjoint[..., 0]
        mu = adjoint[..., 1]
        # A conductance on segment k enters the diagonal (pulling v_k and r_k down) and, times its reversal value,
        # the drive. Hence dLoss/dg = lam_k * (E - v_k) - mu_k * r_k = (-lam_k v_k - mu_k r_k) + E * lam_k.
        grad_conductance = self.at_synapses_affine(-lam * state.v - mu * state.r, lam)
        grad_weight = np.sum(grad_conductance * state.x, axis=1)
        grad_inputs = self.input_sum(grad_conductance * self.weight[:, None, :])
        grads = {"weight": grad_weight, "threshold": grad_threshold, "output_threshold": grad_output_threshold}
        return grads, grad_inputs


    def recirculate(self, state, output_change, trigger_change, send_back):
        """Second pass of recirculation learning: how this layer's neurons respond to changes asked of their outputs.

        output_change (batch, neurons, outputs) or None, and trigger_change (batch, neurons), are the changes of the
        local and trigger outputs requested from further up (by the readout or by later layers). They enter the neurons
        as currents, and the neurons settle once more around their forward state:
        - A requested change of a local output enters the segment of that output, but only through an output synapse
          that is transmitting (boosted value above threshold); a silent output passes nothing back.
        - A requested change of the trigger output enters the trigger, scaled by the trigger's slope p (1 - p) / T,
          since the output follows its trigger value through that slope.
        - Local outputs also follow the trigger output, through the back-boost (each rises by back_boost * z * r_k).
          The trigger therefore adds back_boost * sum_k r_k * current_k to the change requested of its output. Because
          M is symmetric, that sum is exactly how far the currents arriving at the outputs raise the trigger value, so
          the trigger can read it off at its own location.
        The segments then move by shift = M^-1 currents. Synaptic conductances do not change between the passes, so
        the response is exactly linear in the currents, and currents at the trigger spread along r without a new solve.

        Every synapse sees its own segment move. A conductance on segment k pulls that segment towards its reversal
        value E_i, so it supports the shift in proportion to factor_i = (E_i - v_k) * shift_k, returned per synapse.
        Its weight changes in proportion to x_i * factor_i. With send_back, each contact also sends back to its source
        its return strength, equal to its forward weight, times the same factor; summed over the contacts of a source,
        this is the change requested of that source, which the layer below receives as its own output change.
        How conductances shape the spread r itself (and through it the back-boost) is not part of the second pass.
        """
        outputs = self.morphology.output_segments
        slope = state.p * (1.0 - state.p) / self.temperature
        requested = trigger_change
        shift = None
        if output_change is not None:
            output_current = output_change * (state.boosted_out > self.output_threshold)
            requested = requested + self.back_boost * np.sum(output_current * state.r[..., outputs], axis=-1)
            shift = self.morphology.solver.solve(state.diagonal, self.outputs_to_segments(output_current)[..., None])[..., 0]
        trigger_shift = (slope * requested)[..., None] * state.r
        shift = trigger_shift if shift is None else shift + trigger_shift
        factor = self.at_synapses_affine(-state.v * shift, shift)
        returned = self.input_sum(factor * self.weight[:, None, :]) if send_back else None
        return SimpleNamespace(shift=shift, factor=factor, returned=returned)


    def calibrate(self, inputs, context):
        """Data-dependent initialisation from a batch of typical inputs.

        Weights are scaled so the mean synaptic conductance per segment equals ConductanceRatio times the mean leak,
        which places segment values in the middle of their range instead of near zero or near saturation. Thresholds
        are then set from quantiles of the resulting values: each neuron's firing probability exceeds 0.5 on about
        CalibrationTargetRate of the inputs and each local output is active on about OutputActiveFraction of them. The
        firing temperature is set relative to the spread of trigger values across inputs, so that firing probabilities
        are far from 0 and 1 only for inputs near threshold.
        """
        x = self.gather_inputs(inputs)
        mean_conductance = float(np.mean(self.segment_sum(self.weight[:, None, :] * x)))
        target = float(config.ConductanceRatio) * float(np.mean(self.morphology.leaks))
        self.weight *= self.dtype(target / max(mean_conductance, 1e-12))
        self.constrain()
        _, _, v, _ = self.settle(inputs, context)
        trigger_values = v[..., self.morphology.trigger]
        self.threshold[:] = np.quantile(trigger_values, 1.0 - float(config.CalibrationTargetRate), axis=0)
        spread = float(np.std(trigger_values - trigger_values.mean(axis=0)))
        self.temperature = self.temperature_factor * max(spread, 1e-6)
        output_values = v[..., self.morphology.output_segments]
        self.output_threshold[:] = np.quantile(output_values, 1.0 - float(config.OutputActiveFraction), axis=0)
