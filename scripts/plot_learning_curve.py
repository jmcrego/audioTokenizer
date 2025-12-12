
import re
import sys
import matplotlib.pyplot as plt

def parse_training_log(path):
    """
    Parse a training log file and extract:
      - steps
      - train_loss
      - eval_steps
      - eval_loss
      
    Returns a dictionary:
    {
        "steps": [...],
        "train_loss": [...],
        "eval_steps": [...],
        "eval_loss": [...]
    }
    """
    
    # Regex for train and eval lines
    train_re = re.compile(r"Train\s+\[Step\s+(\d+)/\d+.*?\]\s+loss=([0-9.]+)")
    eval_re = re.compile(r"Eval\s+\[Step\s+(\d+)/\d+.*?\]\s+loss=([0-9.]+)")
    
    steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

    with open(path, "r") as f:
        for line in f:
            # Match evaluation lines first (they also contain Step/ loss)
            eval_match = eval_re.search(line)
            if eval_match:
                step = int(eval_match.group(1))
                loss = float(eval_match.group(2))
                eval_steps.append(step)
                eval_loss.append(loss)
                continue
            
            # Match training lines
            train_match = train_re.search(line)
            if train_match:
                step = int(train_match.group(1))
                loss = float(train_match.group(2))
                steps.append(step)
                train_loss.append(loss)
                continue

    return {
        "steps": steps,
        "train_loss": train_loss,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
    }


logs = parse_training_log(sys.argv[1])

plt.figure(figsize=(10,5))
plt.plot(logs['steps'], logs['train_loss'], label="Train Loss", marker='o')
plt.plot(logs['eval_steps'], logs['eval_loss'], label="Eval Loss", marker='x', markersize=10)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.show()
