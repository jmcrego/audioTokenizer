import argparse
import json
import matplotlib.pyplot as plt

def plot_logs(jsonl_path, output_file=None, show_plot=False):
    # Read JSONL into a dict
    logs = {"train": {"steps": [], "loss": []},
            "eval": {"steps": [], "loss": [], "wer": [], "cer": [], "bleu": [], "acc": []}}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            type = entry["type"]
            if type == "run":
                continue
            step = entry["step"]
            if entry["type"] == "train":
                logs["train"]["steps"].append(step)
                logs["train"]["loss"].append(entry["loss"])
            elif entry["type"] == "eval":
                logs["eval"]["steps"].append(step)
                logs["eval"]["loss"].append(entry["loss"])
                logs["eval"]["wer"].append(entry["wer"])
                logs["eval"]["cer"].append(entry["cer"])
                logs["eval"]["bleu"].append(entry["bleu"])
                logs["eval"]["acc"].append(entry["acc"])

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top plot: loss ---
    axs[0].plot(logs["train"]["steps"], logs["train"]["loss"], marker=".", linestyle="None", color="grey", label="train_loss")
    axs[0].plot(logs["eval"]["steps"], logs["eval"]["loss"], marker="o", linestyle="-", color="black", label="eval_loss")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training and Evaluation Loss")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(left=0)

    # --- Bottom plot: wer, cer, bleu ---
    ax1 = axs[1]
    ax2 = ax1.twinx()

    ax1.plot(logs["eval"]["steps"], logs["eval"]["wer"], marker=".", linestyle="-", color="red", label="WER")
    ax1.plot(logs["eval"]["steps"], logs["eval"]["cer"], marker="o", linestyle="-", color="orange", label="CER")
    ax2.plot(logs["eval"]["steps"], logs["eval"]["bleu"], marker="^", linestyle="-", color="blue", label="BLEU")

    ax1.set_ylabel("WER / CER (%)")
    ax2.set_ylabel("BLEU (%)")
    axs[1].set_xlabel("Step")
    axs[1].set_title(f"Evaluation Metrics (Final Lang Acc: {logs['eval']['acc'][-1]:.2f})")

    ax1.grid(True)
    axs[1].set_xlim(left=0)

    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1].legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    if show_plot:
        plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training/evaluation JSONL logs")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to JSONL log file")
    parser.add_argument("--output_file", type=str, default=None, help="Output PNG file path")
    parser.add_argument("--show_plot", action="store_true", help="Show the plot in a GUI window")
    args = parser.parse_args()

    plot_logs(args.jsonl_path, output_file=args.output_file, show_plot=args.show_plot)
