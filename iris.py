"""Fisher's Iris flowers as a sequence of continual-learning tasks.

The dataset holds 150 flowers, 50 of each of three species, each described by four lengths in centimetres: sepal
length, sepal width, petal length and petal width. Ordered by the size of their petals the species are setosa (small,
label 0), versicolor (medium, label 1) and virginica (large, label 2). Setosa is linearly separable from the other two;
versicolor and virginica are not, because a few flowers of each lie among the other.

Tasks
    IrisTasks lists the classes shown by each task, in order. Tasks may differ in size and may share classes, so one
    setting describes strictly class-incremental streams ([[0, 1], [2]]), streams in which earlier classes return
    ([[0, 1], [1, 2], [0, 1, 2]]) and ordinary joint training ([[0, 1, 2]]). The agent always chooses among all three
    classes, as on MNIST, where all ten stay available: the class a task leaves out can still be chosen, and a choice
    of it is simply not rewarded.

Split
    There is no official test set. IrisTestPerClass flowers of every class are held out for testing, drawn from the
    problem's random generator, so every seed gets its own split and the summary over seeds averages over splits as
    well as over initialisations. A task's test set holds the held-out flowers of that task's classes, each measured
    once.

Encoding
    A synapse opens a conductance w * x with x >= 0, and a segment without input relaxes towards zero. A neuron that
    reads the raw lengths therefore grows more active as the flower grows, and can hardly single out a medium value,
    which is exactly what sets versicolor apart. Each length is first scaled to [0, 1] with fixed physical ranges
    (FeatureRanges). The ranges are fixed rather than estimated from the data, so that the flowers of later tasks do
    not shape the inputs of earlier ones. The scaled values are then encoded by IrisEncoding:

    population  IrisUnitsPerFeature Gaussian tuning curves per feature, centres evenly spaced over [0, 1], standard
                deviation IrisTuningWidth times the spacing of the centres. A value excites a few neighbouring units,
                so small, medium and large flowers drive largely different inputs, and learning one class changes fewer
                of the synapses that the other classes rely on.
    complement  every value u together with 1 - u. The total input is the same for every flower, but most units are
                active for most flowers, so all tasks act largely on the same synapses.
    raw         the scaled values u alone.

Observation units have no spatial layout (positions None), so local wiring draws from all of them, as global wiring
does. With ProvideContext the observation of task t comes with the one-hot code of t.

The data file is downloaded once into DataDir. The copies listed in IrisMirrors hold the same 150 flowers (with the
values of Fisher's paper), written either with numeric labels below a header line or with species names; parse_iris
reads both and checks that 50 flowers of each species arrived.
"""

import os
import urllib.request

import numpy as np

import config
from bandit import DatasetBandit
from problem import Problem, ProblemSpec


IrisMirrors = [
    "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/iris.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data",
]
DataFile = "iris.csv"

ClassNames = ["setosa", "versicolor", "virginica"]
NumClasses = len(ClassNames)
FlowersPerClass = 50
# A label may be written as the class number, the species name, or the species name with the "Iris-" prefix.
LabelNames = {key: label for label, name in enumerate(ClassNames) for key in (str(label), name, "Iris-" + name)}

# Lowest and highest value of each feature in centimetres (sepal length, sepal width, petal length, petal width). The
# ranges enclose every flower of the dataset, so the scaled values of the dataset itself lie in [0, 1] without clipping.
FeatureRanges = np.array([[4.0, 8.0], [2.0, 4.5], [1.0, 7.0], [0.0, 2.6]])

Encodings = ["population", "complement", "raw"]


def parse_iris(text):
    """Features (150, 4) in centimetres and labels (150,) from the text of either copy of the data file.

    A flower is a line of four numbers and a label. Blank lines are skipped, as is the header of the scikit-learn copy
    (flower count, feature count and the species names). Anything else stops with an error, and so does a file that
    does not hold exactly 50 flowers of every species, so that a damaged download or an error page is never mistaken
    for data.
    """
    features = []
    labels = []
    for number, line in enumerate(text.splitlines(), start=1):
        fields = [field.strip() for field in line.split(",")]
        if fields == [""] or fields[2:] == ClassNames:
            continue
        if len(fields) != 5 or fields[4] not in LabelNames:
            raise ValueError("Unexpected line %d in the Iris data: %r" % (number, line))
        features.append([float(field) for field in fields[:4]])
        labels.append(LabelNames[fields[4]])
    labels = np.array(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=NumClasses)
    if len(labels) != NumClasses * FlowersPerClass or np.any(counts != FlowersPerClass):
        raise ValueError("The Iris data should hold %d flowers of each species; found %s"
                         % (FlowersPerClass, counts.tolist()))
    return np.array(features, dtype=np.float64), labels


def download(path):
    """Fetch the data file from the first mirror that answers with valid data and store it at path.

    The text is parsed before it is stored, so only a complete and valid file is ever cached.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    errors = []
    for url in IrisMirrors:
        try:
            print("Downloading " + url)
            with urllib.request.urlopen(url, timeout=60) as response:
                text = response.read().decode("utf-8")
            parse_iris(text)
            with open(path + ".part", "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(path + ".part", path)
            return
        except Exception as error:
            errors.append(url + ": " + str(error))
    raise RuntimeError("Could not download the Iris data:\n" + "\n".join(errors))


def load_iris(data_dir):
    """Features (150, 4) and labels (150,) of all flowers, downloading the data file into data_dir on first use."""
    path = os.path.join(data_dir, DataFile)
    if not os.path.exists(path):
        download(path)
    with open(path, "r", encoding="utf-8") as handle:
        return parse_iris(handle.read())


def encode(features, encoding, units_per_feature, tuning_width):
    """Observation vectors (flowers, size) with values in [0, 1] for features (flowers, 4) in centimetres.

    The population code lists the tuning curves of the first feature, then those of the second, and so on. Neighbouring
    centres are one spacing apart, so with a width of one spacing a value at a centre gives its neighbours about 0.61
    and the next ones about 0.14: each value is carried by roughly three units of its feature.
    """
    scaled = np.clip((features - FeatureRanges[:, 0]) / (FeatureRanges[:, 1] - FeatureRanges[:, 0]), 0.0, 1.0)
    if encoding == "raw":
        return scaled
    if encoding == "complement":
        return np.concatenate([scaled, 1.0 - scaled], axis=1)
    if encoding == "population":
        units = int(units_per_feature)
        if units < 2:
            raise ValueError("IrisUnitsPerFeature must be at least 2 for the population encoding")
        centres = np.linspace(0.0, 1.0, units)
        width = float(tuning_width) * (centres[1] - centres[0])
        if width <= 0.0:
            raise ValueError("IrisTuningWidth must be positive")
        distance = (scaled[:, :, None] - centres) / width
        return np.exp(-0.5 * distance ** 2).reshape(len(features), -1)
    raise ValueError("Unknown Iris encoding '" + str(encoding) + "'; known: " + ", ".join(Encodings))


def hold_out(labels, test_per_class, rng):
    """Indices of the training and the test flowers, with test_per_class random flowers of every class held out."""
    if not 1 <= test_per_class < FlowersPerClass:
        raise ValueError("IrisTestPerClass must lie between 1 and %d" % (FlowersPerClass - 1))
    train = []
    test = []
    for label in range(NumClasses):
        members = rng.permutation(np.flatnonzero(labels == label))
        test.append(members[:test_per_class])
        train.append(members[test_per_class:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def check_tasks(tasks):
    """IrisTasks as a list of lists of class numbers, after checking that every task names distinct existing classes."""
    message = "IrisTasks must be a non-empty list of tasks, each a list of distinct classes among 0, 1 and 2; got "
    try:
        result = [[int(label) for label in task] for task in tasks]
    except (TypeError, ValueError):
        raise ValueError(message + repr(tasks))
    if not result:
        raise ValueError(message + repr(tasks))
    for task in result:
        if not task or len(set(task)) != len(task) or min(task) < 0 or max(task) >= NumClasses:
            raise ValueError(message + repr(tasks))
    return result


class IrisIncremental(Problem):
    """Iris as a sequence of tasks over chosen subsets of the three species, each task served as a contextual bandit."""

    def __init__(self, dtype, rng):
        self.dtype = dtype
        self.rng = rng
        self.task_classes = check_tasks(config.IrisTasks)
        self.num_tasks = len(self.task_classes)
        features, labels = load_iris(config.DataDir)
        observations = encode(features, config.IrisEncoding, config.IrisUnitsPerFeature, config.IrisTuningWidth)
        observations = observations.astype(dtype)
        train, test = hold_out(labels, int(config.IrisTestPerClass), rng)
        self.data = {
            "train_items": observations[train],
            "train_labels": labels[train],
            "test_items": observations[test],
            "test_labels": labels[test],
        }
        self.provide_context = bool(config.ProvideContext)
        context_size = self.num_tasks if self.provide_context else 0
        self.spec = ProblemSpec(observations.shape[1], None, context_size, NumClasses)


    def task_name(self, task):
        return "-".join(ClassNames[label] for label in self.task_classes[task])


    def _environment(self, task, batch_size, split, shuffle):
        labels = self.data[split + "_labels"]
        selected = np.flatnonzero(np.isin(labels, self.task_classes[task]))
        context = np.eye(self.num_tasks)[task] if self.provide_context else None
        items = self.data[split + "_items"][selected]
        return DatasetBandit(items, labels[selected], batch_size, shuffle, self.rng, self.dtype, context=context)


    def train_environment(self, task, batch_size):
        return self._environment(task, batch_size, "train", True)


    def test_environment(self, task, batch_size):
        return self._environment(task, batch_size, "test", False)
