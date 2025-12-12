
import os
import torch
import logging
import numpy as np

from AudioToLLMWrapper import AudioToLLMWrapper
from AudioToLLMTrainer import AudioToLLMTrainer
from AudioToLLMDataset import AudioDataset

logger = logging.getLogger("train_sft")

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
    dtype = torch.float32
    return device, dtype    

if __name__ == "__main__":
    import logging
    import argparse

    parser = argparse.ArgumentParser(description="Train a speech ASR/STT decoder (audio-embedder ➔ Projector ➔ LLM).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model paths
    parser.add_argument("--audio_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147")
    parser.add_argument("--proj_path", type=str, default=None)
    parser.add_argument("--llm_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    # dataset paths
    parser.add_argument("--train", required=True, help="Training dataset file")
    parser.add_argument("--eval", default=None, help="Evaluation dataset file")
    # model vars
    parser.add_argument("--chunk_size", type=int, default=3200, help="Group this many samples when building chunks in audio processor")
    parser.add_argument("--stride", type=int, default=1600, help="Overlap this many samples when building chunks in audio processor")
    parser.add_argument("--stack_size", type=int, default=8, help="Stack this many frames in audio to LLM projector")
    parser.add_argument("--rank_dim", type=int, default=256, help="Low-rank intermediate dimension for audio to LLM projector")
    # optimization pars
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--accum_steps", type=int, default=4, help="Accumulate this many steps before optimizing")
    # training pars
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps (must be >0 for scheduler)")
    parser.add_argument("--max_epochs", type=int, default=0, help="Maximum number of training epochs (0 for no limit)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--eval_every", type=int, default=5000, help="Run evaluation after this many steps")
    parser.add_argument("--log_every", type=int, default=500, help="Logging after this many steps")
    # output
    parser.add_argument("--output_dir", type=str, default="./sft_output", help="Output directory of training")
    parser.add_argument("--debug", action="store_true", help="Debug mode with more logging")
    args = parser.parse_args()

    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logging
    log_filename = os.path.join(args.output_dir, f"train.log") #_{datetime.now().strftime('%Y%m%d_%H%M%S')}
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
    # Load model wrapper
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
    )

    # -----------------------------
    # Datasets 
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
    ) if args.eval is not None else None

    # -----------------------------
    # Create Trainer
    # -----------------------------

    trainer = AudioToLLMTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        eval_every=args.eval_every,
        log_every=args.log_every,
        accum_steps=args.accum_steps,
        output_dir=args.output_dir,
        device=device,
        dtype=dtype
    )

    # -----------------------------
    # Start training
    # -----------------------------

    trainer.train()

