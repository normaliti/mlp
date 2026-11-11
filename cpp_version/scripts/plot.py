#!/usr/bin/env python3
import csv
import os
import sys

def read_history(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def to_floats(rows, key):
    return [float(r[key]) for r in rows]

def main():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required. Install with: pip install matplotlib")
        return 1

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    outdir = "models"
    if "--outdir" in sys.argv:
        idx = sys.argv.index("--outdir")
        if idx + 1 < len(sys.argv):
            outdir = sys.argv[idx + 1]

    if not args:
        args = [os.path.join(outdir, "history.csv")]

    histories = []
    for p in args:
        if not os.path.exists(p):
            print("History file not found:", p)
            return 1
        histories.append((os.path.basename(p), read_history(p)))

    os.makedirs(outdir, exist_ok=True)

    plt.figure()
    for name, rows in histories:
        epochs = [int(r["epoch"]) for r in rows]
        plt.plot(epochs, to_floats(rows, "loss"), label=f"{name} train")
        plt.plot(epochs, to_floats(rows, "val_loss"), label=f"{name} valid")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss.png"))
    plt.close()

    plt.figure()
    for name, rows in histories:
        epochs = [int(r["epoch"]) for r in rows]
        plt.plot(epochs, to_floats(rows, "acc"), label=f"{name} train")
        plt.plot(epochs, to_floats(rows, "val_acc"), label=f"{name} valid")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy.png"))
    plt.close()

    print("Saved plots to:", outdir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
