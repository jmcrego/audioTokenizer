import argparse
import json
import matplotlib.pyplot as plt

def read_jsonl(file_path):
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    wer = []
    cer = []
    bleu = []
    lang_acc = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["type"] == "train":
                train_steps.append(entry["step"])
                train_loss.append(entry["loss"])
            elif entry["type"] == "eval":
                eval_steps.append(entry["step"])
                eval_loss.append(entry.get("eval_loss", None))
                wer.append(entry.get("wer", None))
                cer.append(entry.get("cer", None))
                bleu.append(entry.get("bleu", None))
                lang_acc.append(entry.get("lang_acc", None))
    return {
        "train": {"steps": train_steps, "loss": train_loss},
        "eval": {"steps": eval_steps, "loss": eval_loss, "wer": wer, "cer": cer, "bleu": bleu, "lang_acc": lang_acc}
    }

def plot_logs(logs, output_file="training_plot.png", show_plot=True):
    plt.figure(figsize=(12, 8))

    # --- First subplot: train + eval loss ---
    plt.subplot(2, 1, 1)
    plt.plot(logs["train"]["steps"], logs["train"]["loss"], label="train_loss", color="blue")
    plt.plot(logs["eval"]["steps"], logs["eval"]["loss"], label="eval_loss", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)

    # --- Second subplot: WER, CER, BLEU (dual axis) ---
    plt.subplot(2, 1, 2)
    steps = logs["eval"]["steps"]
    fig, ax1 = plt.gcf(), plt.gca()
    ax1.plot(steps, logs["eval"]["wer"], label="WER", color="red", marker="o")
    ax1.plot(steps, logs["eval"]["cer"], label="CER", color="green", marker="x")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("WER / CER (%)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(steps, logs["eval"]["bleu"], label="BLEU", color="blue", linestyle="--", marker="s")
    ax2.set_ylabel("BLEU Score")
    
    # --- Combine legends ---
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    # --- Add final lang_acc label ---
    final_acc = logs["eval"]["lang_acc"][-1] if logs["eval"]["lang_acc"] else None
    if final_acc is not None:
        ax1.text(0.98, 0.95, f"Final lang_acc: {final_acc:.2f}", transform=ax1.transAxes,
                 horizontalalignment="right", verticalalignment="top", bbox=dict(facecolor="white", alpha=0.5))

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        print(f"Saved plot as {output_file}")

    if show_plot:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training/evaluation JSONL logs")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to JSONL log file")
    parser.add_argument("--show_plot", action="store_true", help="Show the plot in a GUI window")
    parser.add_argument("--output_file", type=str, default=None, help="Output PNG file path")
    args = parser.parse_args()

    logs = read_jsonl(args.jsonl_path)
    plot_logs(logs, output_file=args.output_file, show_plot=args.show_plot)
