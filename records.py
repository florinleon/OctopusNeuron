"""Continual-learning results: the accuracy matrix, training curves, summary metrics, files and figure.

The accuracy matrix holds R[t, k], the score on the test set of task k measured right after training on task t. From it:

average_accuracy[t]   mean of R[t, k] over the tasks seen so far (k <= t)
forgetting[k]         best score task k ever had after its own training, minus its final score (k < last task)
learning_speed[t]     mean training reward over the first SpeedWindow steps of task t
final_training[t]     mean training reward over the last SpeedWindow steps of task t

learning_speed answers whether new tasks keep being learned as quickly as the first one (loss of plasticity shows up as
a falling learning speed), forgetting answers whether old tasks survive the new ones.

When the same experiment is run with several seeds, summarize_runs lists the main measures of every run together with
their mean and standard deviation over the runs.
"""

import csv
import json
import os

import numpy as np


# Reference palette: one hue for single-series lines, a light-to-dark ramp of the same hue for magnitudes.
LineColor = "#2a78d6"
BoundaryColor = "#c3c2b7"
SurfaceColor = "#fcfcfb"
TextColor = "#0b0b0b"
MutedTextColor = "#52514e"
SequentialRamp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Columns of the summary over runs: (label, function of one run's results returning a number).
SummaryColumns = [
    ("final accuracy", lambda result: result["final_average_accuracy"]),
    ("mean forgetting", lambda result: result["mean_forgetting"]),
    ("accuracy right after training", lambda result: float(np.mean(np.diag(np.array(result["accuracy_matrix"]))))),
    ("early training reward", lambda result: float(np.mean(result["learning_speed"]))),
]


class ContinualRecord:
    """Collects per-task training curves and evaluation scores and turns them into metrics and files."""

    def __init__(self, task_names, speed_window):
        self.task_names = list(task_names)
        self.num_tasks = len(task_names)
        self.speed_window = int(speed_window)
        self.matrix = np.full((self.num_tasks, self.num_tasks), np.nan)
        self.curves = []


    def add_task(self, task, curve, scores):
        self.curves.append(np.asarray(curve, dtype=float))
        self.matrix[task, :] = scores


    def summary(self):
        trained = len(self.curves)
        average = [float(np.mean(self.matrix[t, : t + 1])) for t in range(trained)]
        forgetting = []
        for task in range(trained - 1):
            forgetting.append(float(np.max(self.matrix[task:trained - 1, task]) - self.matrix[trained - 1, task]))
        window = self.speed_window
        return {
            "task_names": self.task_names[:trained],
            "average_accuracy": average,
            "final_average_accuracy": average[-1] if average else float("nan"),
            "forgetting": forgetting,
            "mean_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
            "learning_speed": [float(np.mean(curve[:window])) for curve in self.curves],
            "final_training_reward": [float(np.mean(curve[-window:])) for curve in self.curves],
            "accuracy_matrix": self.matrix[:trained].tolist(),
        }


    def save(self, folder, settings, seed, elapsed_seconds):
        os.makedirs(folder, exist_ok=True)
        # Every setting of config.py with the value it had in this run. Values JSON cannot store are written as text,
        # so that an unusual value set in main.py cannot lose the results of a long run.
        with open(os.path.join(folder, "config.json"), "w") as handle:
            json.dump(settings, handle, indent=2, default=str)
        result = self.summary()
        result["seed"] = int(seed)
        result["elapsed_seconds"] = float(elapsed_seconds)
        with open(os.path.join(folder, "results.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        with open(os.path.join(folder, "accuracy_matrix.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["after_training"] + self.task_names)
            for task in range(len(self.curves)):
                writer.writerow([self.task_names[task]] + ["%.4f" % value for value in self.matrix[task]])
        with open(os.path.join(folder, "learning_curves.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task", "step", "mean_reward"])
            for task, curve in enumerate(self.curves):
                for step, value in enumerate(curve, start=1):
                    writer.writerow([task, step, "%.4f" % value])
        self._plot(os.path.join(folder, "summary.png"), result)
        return result


    def _plot(self, path, result):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            print("matplotlib is not installed; skipping the summary figure")
            return
        trained = len(self.curves)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.1, 1.0, 1.6]})
        figure.patch.set_facecolor(SurfaceColor)
        for axis in axes:
            axis.set_facecolor(SurfaceColor)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color(BoundaryColor)
            axis.tick_params(colors=MutedTextColor, labelsize=8)

        # Accuracy matrix as a single-hue heatmap; the values are printed in the cells (they double as a table view).
        ramp = LinearSegmentedColormap.from_list("ramp", SequentialRamp)
        matrix = self.matrix[:trained]
        axes[0].imshow(matrix, cmap=ramp, vmin=0.0, vmax=1.0, aspect="auto")
        for row in range(trained):
            for col in range(self.num_tasks):
                value = matrix[row, col]
                ink = "white" if value > 0.6 else TextColor
                axes[0].text(col, row, "%.2f" % value, ha="center", va="center", fontsize=7, color=ink)
        axes[0].set_xticks(range(self.num_tasks))
        axes[0].set_xticklabels([str(k) for k in range(self.num_tasks)])
        axes[0].set_yticks(range(trained))
        axes[0].set_yticklabels([str(t) for t in range(trained)])
        axes[0].set_xlabel("evaluated task", color=MutedTextColor, fontsize=9)
        axes[0].set_ylabel("after training task", color=MutedTextColor, fontsize=9)
        axes[0].set_title("Test accuracy per task", color=TextColor, fontsize=10, loc="left")

        steps = np.arange(trained)
        axes[1].plot(steps, result["average_accuracy"], color=LineColor, linewidth=1.5, marker="o", markersize=5)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xticks(steps)
        axes[1].grid(axis="y", color=BoundaryColor, linewidth=0.5, alpha=0.6)
        axes[1].set_xlabel("after training task", color=MutedTextColor, fontsize=9)
        axes[1].set_title("Average accuracy over seen tasks", color=TextColor, fontsize=10, loc="left")
        axes[1].annotate("%.2f" % result["average_accuracy"][-1], (steps[-1], result["average_accuracy"][-1]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8, color=TextColor)

        # Training reward per step, smoothed, with the task switches marked.
        smoothing = max(1, self.speed_window // 5)
        offset = 0
        for task, curve in enumerate(self.curves):
            kernel = np.ones(min(smoothing, len(curve))) / min(smoothing, len(curve))
            smooth = np.convolve(curve, kernel, mode="valid")
            axes[2].plot(offset + np.arange(len(smooth)) + len(kernel), smooth, color=LineColor, linewidth=1.2)
            if task > 0:
                axes[2].axvline(offset, color=BoundaryColor, linewidth=1.0)
            offset += len(curve)
        axes[2].set_ylim(0.0, 1.0)
        axes[2].grid(axis="y", color=BoundaryColor, linewidth=0.5, alpha=0.6)
        axes[2].set_xlabel("training step (task switches marked)", color=MutedTextColor, fontsize=9)
        axes[2].set_title("Training reward", color=TextColor, fontsize=10, loc="left")
        figure.tight_layout()
        figure.savefig(path, dpi=130, facecolor=SurfaceColor)
        plt.close(figure)


def summarize_runs(folder, seeds, results):
    """Write summary.csv and summary.md to folder and print the table.

    The table lists every measure of SummaryColumns for each run, then its mean and standard deviation over the runs.
    """
    labels = [label for label, _ in SummaryColumns]
    values = np.array([[measure(result) for _, measure in SummaryColumns] for result in results])
    mean = values.mean(axis=0)
    deviation = values.std(axis=0)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run"] + labels)
        for seed, row in zip(seeds, values):
            writer.writerow(["seed%d" % seed] + ["%.4f" % value for value in row])
        writer.writerow(["mean"] + ["%.4f" % value for value in mean])
        writer.writerow(["sd"] + ["%.4f" % value for value in deviation])
    # The markdown table ends with mean ± standard deviation over the runs, ready to paste into notes or a document. It
    # is written as UTF-8 explicitly, because the default encoding of some systems cannot represent the ± sign.
    lines = ["| Run | " + " | ".join(labels) + " |", "|" + " --- |" * (1 + len(labels))]
    for seed, row in zip(seeds, values):
        lines.append("| seed%d | " % seed + " | ".join("%.3f" % value for value in row) + " |")
    lines.append("| mean ± sd | " + " | ".join("%.3f ± %.3f" % pair for pair in zip(mean, deviation)) + " |")
    table = "\n".join(lines) + "\n"
    with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(table)
    print(table)
