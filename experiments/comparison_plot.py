from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def save_comparison_plot(histories, output_path):
    plot_path = Path(output_path)
    plot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    train_loss_axis = axes[0, 0]
    val_loss_axis = axes[0, 1]
    train_accuracy_axis = axes[1, 0]
    val_accuracy_axis = axes[1, 1]

    for history_data in histories:
        model_name = history_data["name"]
        history = history_data["history"]

        epochs = range(
            1,
            len(history["train_loss"]) + 1,
        )

        train_loss_axis.plot(
            epochs,
            history["train_loss"],
            label=model_name,
        )

        val_loss_axis.plot(
            epochs,
            history["val_loss"],
            label=model_name,
        )

        train_accuracy_axis.plot(
            epochs,
            history["train_accuracy"],
            label=model_name,
        )

        val_accuracy_axis.plot(
            epochs,
            history["val_accuracy"],
            label=model_name,
        )

    train_loss_axis.set_title("Training loss")
    val_loss_axis.set_title("Validation loss")
    train_accuracy_axis.set_title("Training accuracy")
    val_accuracy_axis.set_title("Validation accuracy")

    train_loss_axis.set_ylabel("Binary cross-entropy")
    val_loss_axis.set_ylabel("Binary cross-entropy")
    train_accuracy_axis.set_ylabel("Accuracy")
    val_accuracy_axis.set_ylabel("Accuracy")

    train_accuracy_axis.set_ylim(0.0, 1.0)
    val_accuracy_axis.set_ylim(0.0, 1.0)

    for row in axes:
        for axis in row:
            axis.set_xlabel("Epoch")
            axis.grid()
            axis.legend()
            axis.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )

    figure.tight_layout()
    figure.savefig(
        plot_path,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"comparison plot saved to: {plot_path}")