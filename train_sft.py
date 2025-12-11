
import os
import torch
import numpy as np

from trl import SFTTrainer, SFTConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as HFDataset

from AudioToLLMWrapper import AudioToLLMWrapper
from Datasets import AudioDataset, BatchedLengthSampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_device_dtype():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # picks default CUDA device
        try:
            props = torch.cuda.get_device_properties(device)
            name = props.name.lower()
            if "h100" in name:
                dtype = torch.bfloat16
            elif "a100" in name:
                dtype = torch.bfloat16  # optional, you could also use fp16
            else:  # V100, T4, etc.
                dtype = torch.float16
        except Exception:
            dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype

class MySFTTrainer(SFTTrainer):
    def __init__(self, *args, train_loader=None, eval_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_loader = train_loader
        self._eval_loader = eval_loader

    def get_train_dataloader(self):
        return self._train_loader

    def get_eval_dataloader(self, eval_dataset=None):
        return self._eval_loader

    def _prepare_dataset(
        self,
        dataset,
        processing_class=None,
        args=None,
        packing=False,
        formatting_func=None,
        dataset_name=None,
    ):
        return dataset # Skip tokenization since we handle everything in the collator
    

if __name__ == "__main__":
    import logging
    import argparse
    #import subprocess

    parser = argparse.ArgumentParser(description="Train a speech ASR/STT decoder (audio-embedder ➔ Projector ➔ LLM).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model paths
    parser.add_argument("--audio_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147")
    parser.add_argument("--proj_path", type=str, default=None)
    parser.add_argument("--llm_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    # dataset paths
    parser.add_argument("--train", required=True, help="Training dataset file")
    parser.add_argument("--eval", required=True, help="Evaluation dataset file")
    # model vars
    parser.add_argument("--chunk_size", type=int, default=3200, help="Group this many samples when building chunks in audio processor")
    parser.add_argument("--stride", type=int, default=1600, help="Overlap this many samples when building chunks in audio processor")
    parser.add_argument("--stack_size", type=int, default=8, help="Stack this many frames in audio to LLM projector")
    parser.add_argument("--rank_dim", type=int, default=256, help="Low-rank intermediate dimension for audio to LLM projector")
    # optimization pars
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # training pars
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run evaluation after this many steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging after this many steps")

    parser.add_argument("--output_dir", type=str, default="./sft_output", help="Output directory of training")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Log file name with timestamp
    log_filename = os.path.join(args.output_dir, f"train.log") #_{datetime.now().strftime('%Y%m%d_%H%M%S')}

    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),  # save to file
            logging.StreamHandler()  # and print to console
        ]
    )

    logger = logging.getLogger(__name__)
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")

    # -----------------------------
    # 1. Load models wrapper
    # -----------------------------
    model = AudioToLLMWrapper(
        audio_path=args.audio_path,
        proj_path=args.proj_path,
        llm_path=args.llm_path,
        chunk_size=args.chunk_size,
        stride=args.stride,
        stack_size=args.stack_size,
        rank_dim=args.rank_dim,
        max_seq_len=args.max_seq_len,
        device=device,
        dtype=dtype,
    ).to(device, dtype=dtype) #is this .to needed? 

    print("Trainable params in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # -----------------------------
    # 3. Datasets and collator
    # -----------------------------
    train_dataset = AudioDataset(
        file_path=args.train,
        tokenizer=model.tokenizer,
        asr_token="[ASR]",
        stt_token="[STT]",
        end_token="[END]",
        sample_rate=model.audio_embedder.sample_rate,
        chunk_size=args.chunk_size,
        stride=args.stride,
        stack_size=args.stack_size,
        max_seq_len=args.max_seq_len
    )
    eval_dataset = AudioDataset(
        file_path=args.eval,
        tokenizer=model.tokenizer,
        asr_token="[ASR]",
        stt_token="[STT]",
        end_token="[END]",
        sample_rate=model.audio_embedder.sample_rate,
        chunk_size=args.chunk_size,
        stride=args.stride,
        stack_size=args.stack_size,
        max_seq_len=args.max_seq_len
    )

    train_sampler = BatchedLengthSampler(train_dataset, batch_size=args.batch_size)
    eval_sampler = BatchedLengthSampler(eval_dataset, batch_size=args.batch_size)

    # Collator returns audio_paths, prompt_ids, target_ids
    def collator_fn(batch):
        audio_paths = [x["audio_path"] for x in batch]
        prompt_ids = pad_sequence([torch.tensor(x["prompt_ids"]) for x in batch], batch_first=True, padding_value=model.tokenizer.pad_token_id)
        target_ids = pad_sequence([torch.tensor(x["target_ids"]) for x in batch], batch_first=True, padding_value=model.tokenizer.pad_token_id)
        return {
            "audio_paths": audio_paths,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids
        }

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator_fn
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        collate_fn=collator_fn
    )

    # -----------------------------
    # 4. SFTTrainer
    # -----------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        gradient_checkpointing=False,
        dataset_text_field=None,
        dataset_kwargs={"add_special_tokens": False, "map_fn": lambda x: x},
        packing=False,
    )

    trainer = MySFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=HFDataset.from_list([{"text": ""}]),
        eval_dataset=HFDataset.from_list([{"text": ""}]),
        data_collator=collator_fn,
        train_loader=train_loader,
        eval_loader=eval_loader
    )

    batch = next(iter(trainer.get_train_dataloader()))

    print("------ BATCH CONTENT ------")
    print("Audio paths:", batch["audio_paths"])
    print("Prompt IDs:", batch["prompt_ids"].shape)
    print(batch["prompt_ids"])
    print("Target IDs:", batch["target_ids"].shape)
    print(batch["target_ids"])

    print("\n------ MODEL OUTPUT ------")
    outputs = trainer.model(**batch)

    print("Loss:", outputs["loss"])
    print("Logits shape:", outputs["logits"].shape)
    print("Labels shape:", outputs["labels"].shape)
    print("Attention mask shape:", outputs["attention_mask"].shape)

    trainer.train()
