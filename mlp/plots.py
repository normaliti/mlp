from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def save_plots(history, output_dir):
    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    loss_path = plots_dir / "loss.png"
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Binary cross-entropy")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    accuracy_path = plots_dir / "accuracy.png"
    plt.figure()
    plt.plot(epochs, history["train_accuracy"], label="train")
    plt.plot(epochs, history["val_accuracy"], label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(0.0, 1.0)
    plt.grid()
    plt.legend()
    plt.savefig(accuracy_path, bbox_inches="tight")
    plt.close()

    print(f"loss graph saved to: {loss_path}")
    print(f"accuracy graph saved to: {accuracy_path}")
