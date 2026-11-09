"""Spatial layout of neurons and the rules that decide which sources a neuron samples.

Every source (an observation unit, a local output, a trigger output) and every neuron has a position in the unit square.
Positions carry the one structural distinction the network relies on: local outputs only reach neurons nearby, while
trigger outputs may reach any neuron. Problems without a spatial structure can give all observation units the same
position, in which case "local" wiring degenerates gracefully into "global" wiring.
"""

import numpy as np


def grid_positions(rows, cols, neurons_per_position):
    """Positions of a grid of neuron groups, neurons of one position stored next to each other.

    Group centres are placed at the middle of equal tiles of the unit square, so a 7 x 7 grid over a 28 x 28 image
    puts one group on the centre of every 4 x 4 pixel tile.
    """
    centres = [((row + 0.5) / rows, (col + 0.5) / cols) for row in range(rows) for col in range(cols)]
    return np.repeat(np.array(centres, dtype=np.float64), neurons_per_position, axis=0)


def sample_sources(neuron_positions, source_positions, spec, rng):
    """Choose spec["fan_in"] source indices for every neuron; returns an integer array (num_neurons, fan_in).

    kind "local": candidates are sources whose position lies inside the square of half-width spec["radius"] around the
    neuron. kind "global": every source is a candidate. Candidates are drawn without replacement when there are enough
    of them; otherwise all of them are taken and the rest is drawn with replacement, which places the same source on
    several synapses of the neuron (a source can then reach several branches, as repeated contacts do in real neurons).
    """
    fan_in = int(spec["fan_in"])
    num_sources = len(source_positions)
    chosen = np.empty((len(neuron_positions), fan_in), dtype=np.int64)
    for neuron, position in enumerate(neuron_positions):
        if spec["kind"] == "local":
            distance = np.max(np.abs(source_positions - position), axis=1)
            candidates = np.flatnonzero(distance <= float(spec["radius"]))
            if len(candidates) == 0:
                candidates = np.argsort(distance)[:fan_in]
        elif spec["kind"] == "global":
            candidates = np.arange(num_sources)
        else:
            raise ValueError("Unknown wiring kind '" + str(spec["kind"]) + "'")
        if len(candidates) >= fan_in:
            chosen[neuron] = rng.choice(candidates, size=fan_in, replace=False)
        else:
            extra = rng.choice(candidates, size=fan_in - len(candidates), replace=True)
            chosen[neuron] = np.concatenate([candidates, extra])
    return chosen
