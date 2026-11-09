"""Shape of a unipolar neuron and the solver for its steady state.

A neuron is a tree of K segments. Every segment k loses activity through a leak l_k, and two adjacent segments are
joined by a coupling c (a thick wire has a large coupling). Synapses add a conductance g to the segment they sit on and,
for excitatory synapses, a drive g * E pulling the segment towards the reversal value E. The steady state v solves

    (diag(l) + L + diag(g_total)) v = drive,

where L is the weighted graph Laplacian of the tree. The matrix is symmetric, positive definite and tree-structured,
so it can be solved exactly in O(K) operations by eliminating segments from the leaves towards the root and then
substituting back from the root towards the leaves (the Hines algorithm used by compartmental neuron simulators).
The same matrix appears in the backward pass, because it is symmetric: errors spread through a neuron by the same rule
as activity.
"""

import numpy as np

import config


class Morphology:
    """Fixed geometry of a neuron: segments, tree edges, leaks, trigger, output segments and gate segments.

    The tree is rooted at the trigger segment. parents[k] is the neighbour of segment k on the path towards the
    trigger (-1 for the trigger itself) and couplings[k] is the coupling between k and parents[k]. Gate segments are
    the places where context inhibition can be applied; in the star shape they are the entry segments of the branches,
    so inhibiting one of them isolates the whole branch from the trunk.
    """

    def __init__(self):
        if config.MorphologyType == "star":
            self._build_star()
        elif config.MorphologyType == "custom":
            self._build_custom()
        else:
            raise ValueError("Unknown morphology type '" + str(config.MorphologyType) + "'")
        self.num_segments = len(self.parents)
        self.num_outputs = len(self.output_segments)
        self._check_tree()
        # Diagonal of the matrix without synapses: own leak plus the couplings to all neighbours.
        self.base_diagonal = self.leaks.copy()
        for segment, parent in enumerate(self.parents):
            if parent >= 0:
                self.base_diagonal[segment] += self.couplings[segment]
                self.base_diagonal[parent] += self.couplings[segment]
        # Output index of every segment (-1 for segments without local output synapses).
        self.output_index = -np.ones(self.num_segments, dtype=np.int64)
        self.output_index[self.output_segments] = np.arange(self.num_outputs)
        self.solver = TreeSolver(self.parents, self.couplings)


    def _build_star(self):
        branches = int(config.Branches)
        length = int(config.SegmentsPerBranch)
        parents = [-1]
        couplings = [0.0]
        leaks = [float(config.TrunkLeak)]
        tips = []
        entries = []
        for branch in range(branches):
            for position in range(length):
                segment = 1 + branch * length + position
                parents.append(0 if position == 0 else segment - 1)
                couplings.append(float(config.TrunkCoupling if position == 0 else config.BranchCoupling))
                leaks.append(float(config.BranchLeak))
                if position == 0:
                    entries.append(segment)
                if position == length - 1:
                    tips.append(segment)
        self.parents = np.array(parents, dtype=np.int64)
        self.couplings = np.array(couplings, dtype=np.float64)
        self.leaks = np.array(leaks, dtype=np.float64)
        self.trigger = 0
        if config.OutputPlacement == "all_branch":
            self.output_segments = np.arange(1, len(parents), dtype=np.int64)
        elif config.OutputPlacement == "tips":
            self.output_segments = np.array(tips, dtype=np.int64)
        else:
            raise ValueError("Unknown output placement '" + str(config.OutputPlacement) + "'")
        self.gate_segments = np.array(entries, dtype=np.int64)


    def _build_custom(self):
        self.parents = np.array(config.CustomParents, dtype=np.int64)
        self.couplings = np.array(config.CustomCouplings, dtype=np.float64)
        self.leaks = np.array(config.CustomLeaks, dtype=np.float64)
        roots = np.flatnonzero(self.parents < 0)
        if len(roots) != 1:
            raise ValueError("A custom morphology needs exactly one segment with parent -1 (the trigger)")
        self.trigger = int(roots[0])
        self.output_segments = np.array(config.CustomOutputSegments, dtype=np.int64)
        self.gate_segments = np.array(config.CustomGateSegments, dtype=np.int64)


    def _check_tree(self):
        if not (len(self.couplings) == len(self.leaks) == self.num_segments):
            raise ValueError("parents, couplings and leaks must have one entry per segment")
        if np.any(self.leaks <= 0):
            raise ValueError("Every leak must be positive, otherwise the steady state is not unique")
        if np.any(self.output_segments == self.trigger):
            raise ValueError("The trigger segment cannot carry local outputs; its output is the global one")
        # Following parents from any segment must reach the trigger without cycles.
        for segment in range(self.num_segments):
            seen = 0
            node = segment
            while node != self.trigger:
                node = self.parents[node]
                seen += 1
                if node < 0 or seen > self.num_segments:
                    raise ValueError("Segment " + str(segment) + " is not connected to the trigger")


    def dense_matrix(self, extra_diagonal):
        """Full K x K matrix with extra_diagonal added to the diagonal (used only by tests and diagnostics)."""
        matrix = np.diag(self.base_diagonal + extra_diagonal)
        for segment, parent in enumerate(self.parents):
            if parent >= 0:
                matrix[segment, parent] -= self.couplings[segment]
                matrix[parent, segment] -= self.couplings[segment]
        return matrix


class TreeSolver:
    """Exact solver for symmetric tree-structured systems with fixed off-diagonal couplings.

    Only the diagonal changes from one input to the next (synapses add conductance to their segment), so the solver
    takes the diagonal and any number of right-hand sides, batched over arbitrary leading dimensions. Off-diagonal
    entries are -couplings[k] between k and parents[k].
    """

    def __init__(self, parents, couplings):
        self.parents = [int(parent) for parent in parents]
        # Plain Python floats keep the batch arrays in their own precision (a NumPy float64 scalar would upcast
        # float32 arrays to float64 on every multiplication).
        self.couplings = [float(coupling) for coupling in couplings]
        parents = np.asarray(parents)
        depth = np.zeros(len(parents), dtype=np.int64)
        for segment in range(len(parents)):
            node = segment
            while parents[node] >= 0:
                node = parents[node]
                depth[segment] += 1
        self.root = int(np.flatnonzero(parents < 0)[0])
        # Deepest segments first for elimination, shallowest first for back-substitution.
        order = np.argsort(-depth, kind="stable")
        self.upward = [int(k) for k in order if parents[k] >= 0]
        self.downward = self.upward[::-1]


    def solve(self, diagonal, rhs):
        """Solve the system for diagonal of shape (..., K) and rhs of shape (..., K, R); returns (..., K, R)."""
        # Segment-major copies make every per-segment slice contiguous, which matters because the loops below
        # touch one segment at a time across the whole batch.
        d = np.moveaxis(diagonal, -1, 0).copy()
        b = np.moveaxis(rhs, -2, 0).copy()
        for segment in self.upward:
            parent = self.parents[segment]
            coupling = self.couplings[segment]
            factor = coupling / d[segment]
            d[parent] -= coupling * factor
            b[parent] += factor[..., None] * b[segment]
        x = np.empty_like(b)
        x[self.root] = b[self.root] / d[self.root][..., None]
        for segment in self.downward:
            parent = self.parents[segment]
            x[segment] = (b[segment] + self.couplings[segment] * x[parent]) / d[segment][..., None]
        return np.moveaxis(x, 0, -2)
