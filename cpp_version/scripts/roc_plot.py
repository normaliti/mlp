#!/usr/bin/env python3
import csv
import os
import sys

def read_roc(path):
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
        print("Usage: roc_plot.py <roc.csv> [--outdir DIR]")
        return 1

    path = args[0]
    if not os.path.exists(path):
        print("ROC file not found:", path)
        return 1

    rows = read_roc(path)
    fpr = to_floats(rows, "fpr")
    tpr = to_floats(rows, "tpr")

    os.makedirs(outdir, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc.png"))
    plt.close()

    print("Saved ROC plot to:", outdir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
