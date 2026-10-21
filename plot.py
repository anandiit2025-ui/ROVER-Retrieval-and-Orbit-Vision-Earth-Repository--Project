import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metrics(path):
    path = Path(path)
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    steps = [int(row["step"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    valid = [(int(row["step"]), float(row["validation_loss"]))
             for row in rows if row["validation_loss"]]
    memory = [float(row["gpu_reserved_gb"]) for row in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, train, label="Training loss")
    if valid:
        plt.plot([step for step, _ in valid], [loss for _, loss in valid],
                 "o-", label="Validation loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.parent / "loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(steps, memory)
    plt.xlabel("Optimizer step")
    plt.ylabel("Peak PyTorch GPU memory reserved (GiB)")
    plt.tight_layout()
    plt.savefig(path.parent / "gpu_memory.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    plot_metrics(parser.parse_args().metrics)
