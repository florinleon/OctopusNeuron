# Unipolar Neurons for Reward-Based Continual Learning

The project studies networks of unipolar neurons and a local learning rule that trains them from reward alone. Unipolar neurons are the predominant type of nerve cell in the octopus and in most other invertebrates. In a unipolar neuron, a single neurite branches into a tree that carries both input and output synapses. The branches send graded local outputs to nearby cells, while one site on the main neurite sends a global output to distant targets. The network is applied to continual learning, in which a sequence of tasks is learned without access to the data of previous tasks.

## Overview

Each neuron is modeled as a tree of passive segments, whose activations are computed exactly as the steady state of a linear system. Every input source makes one excitatory and one inhibitory synapse on different segments. Excitation saturates and inhibition is shunting, or divisive, so the effect of an input depends on where it is placed on the tree. As a result, a single neuron can compute nonlinear functions such as XOR, which a perceptron cannot.

The network learns without labels: at each step, it chooses an action and receives a reward that indicates whether the action was correct. Learning uses a two-pass rule called recirculation. The first pass computes the outputs and selects an action. The second pass sends requested changes of the outputs back through the same connections, and each synapse adjusts its weight using only quantities available at its own location. Because the system of each neuron is symmetric, this update closely follows the gradient of the error in the predicted reward. Gradient-based training of the same network is included as a reference.

Two mechanisms limit forgetting when tasks are learned in sequence. A task code inhibits a different subset of branches in each task, so that each task uses partly distinct branches and the changes requested by one task barely reach the branches used by others. Consolidation lowers the learning rate of synapses that changed strongly in the recent past. Neither mechanism requires stored data or copies of earlier weights.

## Main Capabilities

- Unipolar neuron model with graded local outputs, a single global output and shunting inhibition.
- Exact computation of the steady state of each neuron, with a solver whose cost is linear in the number of segments.
- Recirculation, a local two-pass learning rule driven only by reward.
- Gradient-based reference learners on the same network, with supervised, reward-only and actor-critic objectives.
- Continual-learning mechanisms based on context-dependent dendritic inhibition and synapse-level consolidation.
- Reward-based task sequences built from labelled datasets presented as contextual bandits, including class-incremental and permuted MNIST and the Iris dataset.
- Evaluation of accuracy across tasks, forgetting and learning speed, with summaries over repeated seeded runs.
- A modular design in which the model and learning rules are independent of the problems, so that new task streams can be added without changes to the network.

## Typical Workflow

A typical experimental workflow consists of:

1. Configuring the model, learning and problem parameters in `config.py`, which defines every setting with its default value and an explanation.
2. Selecting an experiment in `main.py`, which overrides only the settings that differ from the defaults, and running it with `python main.py`.
3. Training the network on a sequence of tasks with recirculation or with a gradient-based reference learner, over multiple random seeds.
4. Evaluating test accuracy on all tasks after each task is learned.
5. Comparing configurations through final accuracy, forgetting and learning speed, using the per-run results and the summaries over runs.

## Citation

A detailed description of the model, the learning rule and the experiments can be found in this paper:

> Florin Leon, *Unipolar Neuron Networks Trained by Recirculation for Reward-Based Continual Learning*, 2026, https://arxiv.org/abs/2611.14153 .

## Note

The implementations are intended as reference programs rather than optimized systems. The programs are distributed in the hope that they will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose.
