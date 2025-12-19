# Trainer.py

import os
import re
import glob
import json
import torch
import shutil
import random
import logging
import numpy as np
from datetime import datetime

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
#from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from Dataset import BatchedLengthSampler 

logger = logging.getLogger("Trainer")

class Color: #for logging
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


class Trainer:
    def __init__(
        self,
        config,
        model,
        train_dataset,
        eval_dataset=None,
        batch_size=4,
        lr_proj=5e-4,
        lr_lora=1e-4,
        max_steps=10000,
        max_epochs=10,
        warmup_steps=0,
        save_best_n=3,
        eval_every=1000,
        log_every=50,
        accum_steps=1,
        output_dir="./output_dir",
        seed=42,
    ):
        
        meta = {k: v for k, v in locals().items() if k != "self" and k != "__class__" and not k.endswith('dataset') and not k == "model"}
        logger.info(f"Initializing {meta}")        

        self.seed_everything(seed)

        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.lr_proj = lr_proj
        self.lr_lora = lr_lora
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.save_best_n = save_best_n
        self.eval_every = eval_every
        self.log_every = log_every
        self.accum_steps = accum_steps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Trainable params in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        param = next(self.model.llm_model.parameters())
        self.device = param.device
        self.dtype = param.dtype

        # -----------------------
        # Sampler & DataLoader
        # -----------------------
        self.train_sampler = BatchedLengthSampler(train_dataset, batch_size=batch_size)
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.collate_fn
        )
        logger.info(f"Initialized Sampler and DataLoader for train with batch_size={batch_size} with {len(self.train_dataset)} samples")

        if eval_dataset is not None:
            self.eval_sampler = BatchedLengthSampler(eval_dataset, batch_size=batch_size)
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_sampler=self.eval_sampler,
                collate_fn=self.collate_fn
            )
            logger.info(f"Initialized Sampler and DataLoader for eval with batch_size={batch_size} with {len(self.train_dataset)} samples")
        else:
            self.eval_loader = None

        if max_epochs:
            self.max_steps = min(self.max_steps, int(len(train_dataset) / (batch_size * accum_steps)))
            logger.info(f"max_steps set to {self.max_steps}")

        # -----------------------
        # Optimizer & Scheduler
        # -----------------------

        self.optimizer = torch.optim.AdamW([
            {"params": self.model.projector.parameters(), "lr": lr_proj},
            {"params": [p for n, p in self.model.llm_model.named_parameters() if p.requires_grad], "lr": lr_lora},
        ])
        logger.info(f"Initialized AdamW optimizer with lr_proj={lr_proj} lr_lora={lr_lora}")

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(self.warmup_steps), num_training_steps=self.max_steps)
        logger.info(f"Initialized Linear scheduler with warmup for {self.max_steps} steps ({self.warmup_steps)} warmup steps)")

        # For logging
        self.step = 0 # optimizer step
        self.batch = 0 # microbatch step
        self.epoch = 0
        self.sample = 0
        self.start_time = datetime.now()


    # -----------------------------
    # Seed everything
    # -----------------------------
    @staticmethod
    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -----------------------
    # Save checkpoint
    # -----------------------
    def save_checkpoint(self, step=None, prefix="checkpoint"):
        step_str = f"_step{step}" if step is not None else ""
        ckpt_path = os.path.join(self.output_dir, f"{prefix}{step_str}")

        # Save Projector
        torch.save(self.model.projector.state_dict(), ckpt_path + ".proj.pt")
        logger.info(f"Saved Projector to {ckpt_path}.proj.pt")

        # Save model LoRa adapters (PEFT)
        self.model.llm_model.save_pretrained(ckpt_path + ".lora")
        logger.info(f"Saved LoRa adapters to {ckpt_path}.lora")

        # save optimizer state (ckpt_path.optim.pt)
        state = {"optimizer_state_dict": self.optimizer.state_dict(), "step": self.step}
        torch.save(state, f"{ckpt_path}.optim.pt")
        print(f"Saved checkpoint to {ckpt_path}")

        # Save config file after updating lora path
        self.config['lora']['path'] = ckpt_path + ".lora"
        self.config['projector']['path'] = ckpt_path + ".proj.pt"
        with open(f"{ckpt_path}.config.json", "w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=4)
        logger.info(f"Saved config to {ckpt_path}.config.json")

        # remove older checkpoints, keep only top N
        remove_old_checkpoints(step, self.output_dir, prefix, self.save_best_n)

    # -----------------------
    # Collator function
    # -----------------------
    def collate_fn(self, batch):
        pad_token_id = self.model.tokenizer.pad_token_id
        audio_paths = [x["audio_path"] for x in batch]
        def ensure_tensor(x):
            return x.detach().clone() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
        prompt_ids = pad_sequence([ensure_tensor(x["prompt_ids"]) for x in batch], batch_first=True, padding_value=pad_token_id)
        target_ids = pad_sequence([ensure_tensor(x["target_ids"]) for x in batch], batch_first=True, padding_value=pad_token_id)
        return {
            "audio_paths": audio_paths,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids
        }

    # -----------------------
    # Logging helper
    # -----------------------
    def log_fn(self, loss, step, epoch, is_eval=False):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        lr_proj = self.optimizer.param_groups[0]["lr"]
        lr_lora = self.optimizer.param_groups[1]["lr"]

        w_step = len(str(self.max_steps))
        w_epoch = len(str(self.max_epochs))

        log_str = (
            f"{'Eval' if is_eval else 'Train'} [Step {Color.CYAN}{step:>{w_step}d}{Color.RESET}/{self.max_steps}, "
            f"Epoch {Color.CYAN}{self.sample/len(self.train_dataset):.3f}{Color.RESET}/{self.max_epochs}] "
            # f"Epoch {Color.CYAN}{epoch:>{w_epoch}d}{Color.RESET}/{self.max_epochs}] "
            f"loss={Color.RED}{loss:.4f}{Color.RESET} | "
            f"lr_proj={Color.GREEN}{lr_proj:.6e}{Color.RESET}, "
            f"lr_lora={Color.GREEN}{lr_lora:.6e}{Color.RESET} | "
            f"elapsed={Color.MAGENTA}{h:02d}h:{m:02d}m:{s:02d}s{Color.RESET}"
        )
        print(log_str)

        log_str = (
            f"{'Eval ' if is_eval else 'Train'} [Step {step:>{w_step}d}/{self.max_steps}, "
            f"Epoch {self.sample/len(self.train_dataset):.3f}/{self.max_epochs}] "
            # f"Epoch {epoch:>{w_epoch}d}/{self.max_epochs}] "
            f"loss={loss:.4f} | "
            f"lr_proj={lr_proj:.6e}, "
            f"lr_lora={lr_lora:.6e} | "
            f"elapsed={h:02d}h:{m:02d}m:{s:02d}s"
        )
        logger.info(log_str)

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
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
            total_loss += outputs["loss"].item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        self.log_fn(avg_loss, is_eval=True)

        self.model.train()
        return avg_loss

    # -----------------------
    # Training loop
    # -----------------------
    def train(self):
        logger.info("Start training")

        self.model.train()
        optimizer = self.optimizer
        optimizer.zero_grad()
        accum_loss = 0.0

        while self.max_steps and self.step < self.max_steps:
            self.epoch += 1            

            for batch in self.train_loader:
                self.batch += 1
                self.sample += batch["prompt_ids"].size(0)
                # Move tensors to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass
                # this with disables automatic mixed precision for everything inside that context.
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    outputs = self.model(**batch)
                    # loss = outputs["loss"] / self.accum_steps  # normalize by accumulation steps
                    raw_loss = outputs["loss"]
                    loss = raw_loss / self.accum_steps                    
                    accum_loss += raw_loss.detach()

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if self.step % self.accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
                    # Scheduler step
                    self.scheduler.step()

                    # Logging (log_every must be multiple of accum_steps)
                    self.step += 1 
                    if self.step % self.log_every == 0:
                        avg_loss = accum_loss / self.accum_steps
                        self.log_fn(avg_loss.item())
                    accum_loss = 0.0

                    # Evaluation + checkpoint
                    if self.eval_loader is not None and self.step % self.eval_every == 0:
                        self.evaluate()
                        self.save_checkpoint(self.step)

                    if self.max_steps and self.step >= self.max_steps:
                        print(f"Reached max steps {self.max_steps}, stopping training after "
                              f"{self.sample} samples, "
                              f"{self.step} steps, "
                              f"{self.batch} batches, "
                              f"{self.sample/len(self.train_dataset):.3f} epochs.")
                        break

            if self.max_epochs and self.epoch >= self.max_epochs:
                print(f"Reached max epochs {self.max_epochs}, stopping training after "
                      f"{self.sample} samples, "
                      f"{self.step} steps, "
                      f"{self.batch} batches, "
                      f"{self.sample/len(self.train_dataset):.3f} epochs.")
                break

        logger.info("End training")

    # -----------------------
    # Logging helper
    # -----------------------
    def log_fn(self, loss, is_eval=False):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        lr_proj = self.optimizer.param_groups[0]["lr"]
        lr_lora = self.optimizer.param_groups[1]["lr"]

        w_step = len(str(self.max_steps))

        log_str = (
            f"{'Eval' if is_eval else 'Train'} "
            f"[Step {Color.CYAN}{self.step:>{w_step}d}{Color.RESET}/{self.max_steps}, "
            f"Epoch {Color.CYAN}{self.sample/len(self.train_dataset):.3f}{Color.RESET}/{self.max_epochs}] "
            f"loss={Color.RED}{loss:.4f}{Color.RESET} | "
            f"lr_proj={Color.GREEN}{lr_proj:.6e}{Color.RESET}, "
            f"lr_lora={Color.GREEN}{lr_lora:.6e}{Color.RESET} | "
            f"elapsed={Color.MAGENTA}{h:02d}h:{m:02d}m:{s:02d}s{Color.RESET}"
        )
        print(log_str)

        log_str = (
            f"{'Eval ' if is_eval else 'Train'} "
            f"[Step {self.step:>{w_step}d}/{self.max_steps}, "
            f"Epoch {self.sample/len(self.train_dataset):.3f}/{self.max_epochs}] "
            f"loss={loss:.4f} | "
            f"lr_proj={lr_proj:.6e}, "
            f"lr_lora={lr_lora:.6e} | "
            f"elapsed={h:02d}h:{m:02d}m:{s:02d}s"
        )
        logger.info(log_str)


def remove_old_checkpoints(step, output_dir, prefix, save_best_n):
    if step is None:
        return

    #Ex: checkpoint_step20000.proj.pt
    existing_steps = []
    for fname in os.listdir(output_dir):
        if fname.startswith(prefix) and fname.endswith(".proj.pt"):
            m = re.search(r"_step(\d+).proj.pt", fname)
            if m:
                existing_steps.append(int(m.group(1)))

    for old_step in sorted(existing_steps, reverse=True)[save_best_n:]:
        old_ckpt_path = os.path.join(output_dir, f"{prefix}_step{old_step}")

        try:
            for path in glob.glob(f"{old_ckpt_path}.*"):
                logger.info(f"Removing {path}.*")
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        except Exception as e:
            print(f"Error removing old checkpoint {old_ckpt_path}: {e}")
