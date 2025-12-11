import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from datetime import datetime

from Datasets import BatchedLengthSampler  # your custom sampler

class AudioToLLMTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset=None,
        batch_size=4,
        lr=1e-4,
        max_steps=10000,
        eval_every=1000,
        log_every=50,
        accum_steps=1,
        output_dir="./output",
        device=None,
        dtype=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.log_every = log_every
        self.accum_steps = accum_steps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("Trainable params in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.model.to(self.device)

        # Optimizer (only trainable params)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )

        # -----------------------
        # Sampler & DataLoader
        # -----------------------

        self.train_sampler = BatchedLengthSampler(train_dataset, batch_size=self.batch_size)
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.collate_fn
        )

        if eval_dataset is not None:
            self.eval_sampler = BatchedLengthSampler(eval_dataset, batch_size=self.batch_size)
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_sampler=self.eval_sampler,
                collate_fn=self.collate_fn
            )
        else:
            self.eval_loader = None

        # For logging
        self.step = 0
        self.start_time = datetime.now()

    # -----------------------
    # Save checkpoint
    # -----------------------
    def save_checkpoint(self, step=None, prefix="checkpoint"):
        step_str = f"_step{step}" if step is not None else ""
        ckpt_path = os.path.join(self.output_dir, f"{prefix}{step_str}.pt")
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step
        }
        torch.save(state, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
        return ckpt_path

    # -----------------------
    # Load checkpoint
    # -----------------------
    def load_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.step = state.get("step", 0)
        print(f"Loaded checkpoint from {ckpt_path}, resuming at step {self.step}")

    # -----------------------
    # Resume from latest checkpoint automatically
    # -----------------------
    def resume_latest_checkpoint(self, prefix="checkpoint"):
        # find all checkpoints with the given prefix
        files = [f for f in os.listdir(self.output_dir) if f.startswith(prefix) and f.endswith(".pt")]
        if not files:
            print("No checkpoint found, starting from scratch.")
            return False
        # pick the one with the largest step number
        def step_from_name(f):
            import re
            m = re.search(r"_step(\d+)", f)
            return int(m.group(1)) if m else 0
        latest_ckpt = max(files, key=step_from_name)
        self.load_checkpoint(os.path.join(self.output_dir, latest_ckpt))
        return True

    # -----------------------
    # Collator function
    # -----------------------
    def collate_fn(self, batch):
        pad_token_id = self.model.tokenizer.pad_token_id
        audio_paths = [x["audio_path"] for x in batch]
        prompt_ids = pad_sequence([torch.tensor(x["prompt_ids"]) for x in batch], batch_first=True, padding_value=pad_token_id)
        target_ids = pad_sequence([torch.tensor(x["target_ids"]) for x in batch], batch_first=True, padding_value=pad_token_id)
        return {
            "audio_paths": audio_paths,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids
        }

    # -----------------------
    # Logging helper
    # -----------------------
    def log_fn(self, loss, step):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"[Step {step}] loss={loss:.4f} | elapsed={elapsed:.1f}s")

    # -----------------------
    # Evaluation
    # -----------------------
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in self.eval_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs["loss"].item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        print(f"=== Eval: avg_loss={avg_loss:.4f}")
        self.model.train()
        return avg_loss

    # -----------------------
    # Training loop
    # -----------------------
    def train(self):
        self.model.train()
        optimizer = self.optimizer
        optimizer.zero_grad()
        for batch in self.train_loader:
            self.step += 1
            # Move tensors to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs["loss"] / self.accum_steps
            loss.backward()

            if self.step % self.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if self.step % self.log_every == 0:
                self.log_fn(loss.item() * self.accum_steps, self.step)

            if self.eval_loader is not None and self.step % self.eval_every == 0:
                self.evaluate()

            if self.step >= self.max_steps:
                break




    # -----------------------------
    # Inspect first batch
    # -----------------------------

    first_batch = next(iter(train_loader))
    print("First batch keys:", first_batch.keys())
    print("Prompt_ids shape:", first_batch["prompt_ids"].shape)
    print("Target_ids shape:", first_batch["target_ids"].shape)
    print("Audio paths (first 5):", first_batch["audio_paths"][:5])
