
import os
import torch
import logging
import argparse
import subprocess
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from AudioEmbedder import AudioEmbedder
from AudioToLLMProjector import AudioToLLMProjector
from datasets import AudioDataset, AudioDatasetIterable

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return commit
    except Exception:
        return "unknown"

def get_device_dtype(): 
    if torch.cuda.is_available():
        device = "cuda:0"
        try:
            props = torch.cuda.get_device_properties(device)
            llm_dtype = torch.bfloat16 if "H100" in props.name else torch.float16
        except Exception:
            llm_dtype = torch.float16
    else:
        device = "cpu"
        llm_dtype = torch.float32
    return device, llm_dtype

# ============================================================
# Compose batch of embeddings/targets with right-padding (vectorized)
# ============================================================
def compose_full_embeddings_with_padding_vectorized(
    proj_embs: torch.Tensor,  # [B, S, D]  (audio embeddings, right-padded)
    audio_mask: torch.Tensor, # [B, S]     (1=real, 0=pad)
    text_embs: torch.Tensor,  # [B, T, D]  (text embeddings, right-padded)
    prompt_ids: torch.Tensor, # [B, T]     (for detecting padding)
    target_ids: torch.Tensor, # [B, L]     (padded with pad_token_id)
    device,
    dtype,
    max_seq_len: int,
    pad_token_id: int,
    ignore_index: int = -100,
):
    """
    Concatenate right-padded audio + prompt embeddings right-padded (already dropped first token)
    and return final padded input_embeds and labels.
    
    Returns:
        input_embeds: [B, L_final, D]
        labels:       [B, L_final]
    """

    B, S, D = proj_embs.shape
    _, T, _ = text_embs.shape
    _, L = target_ids.shape

    # ----------------------------
    # Masks and lengths
    # ----------------------------
    prompt_mask = (prompt_ids != pad_token_id)  # [B, T]
    prompt_lens = prompt_mask.sum(dim=1)        # [B]

    audio_lens = audio_mask.sum(dim=1)          # [B]
    combined_lens = audio_lens + prompt_lens    # [B]
    max_combined_len = min(combined_lens.max().item(), max_seq_len)

    # ----------------------------
    # Prepare output embeddings
    # ----------------------------
    out_embs = torch.zeros((B, max_combined_len, D), device=device, dtype=dtype)

    # ----------------------------
    # Copy audio embeddings
    # ----------------------------
    max_audio_len = S
    audio_idx = torch.arange(max_audio_len, device=device).unsqueeze(0).expand(B, -1)
    audio_valid = audio_idx < audio_lens.unsqueeze(1)
    #(use better the next 3 lines) out_embs[:, :S, :].masked_scatter_(audio_valid.unsqueeze(-1), proj_embs[audio_valid])
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, S)[audio_valid]
    time_idx  = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)[audio_valid]
    out_embs[batch_idx, time_idx, :] = proj_embs[audio_valid]

    # ----------------------------
    # Copy prompt embeddings after audio
    # ----------------------------
    max_prompt_len = text_embs.shape[1]
    prompt_idx = torch.arange(max_prompt_len, device=device).unsqueeze(0).expand(B, -1)
    prompt_valid = prompt_idx < prompt_lens.unsqueeze(1)

    dest_positions = audio_lens.unsqueeze(1) + prompt_idx
    dest_positions = torch.clamp(dest_positions, max=max_combined_len - 1)

    flat_dest = dest_positions[prompt_valid]
    flat_batch = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_prompt_len)[prompt_valid]
    flat_values = text_embs[prompt_valid]

    out_embs[flat_batch, flat_dest, :] = flat_values

    # ----------------------------
    # Build labels (right-aligned targets)
    # ----------------------------
    labels = torch.full((B, max_combined_len), ignore_index, dtype=torch.long, device=device)

    tgt_mask = target_ids != pad_token_id
    tgt_lens = tgt_mask.sum(dim=1) # [B]
    max_tgt_len = tgt_lens.max().item()

    tgt_idx = torch.arange(max_tgt_len, device=device).unsqueeze(0).expand(B, -1)
    tgt_valid = tgt_idx < tgt_lens.unsqueeze(1)

    dest_tgt_pos = max_combined_len - tgt_lens.unsqueeze(1) + tgt_idx
    dest_tgt_pos = torch.clamp(dest_tgt_pos, max=max_combined_len - 1)

    flat_dest_tgt = dest_tgt_pos[tgt_valid]
    flat_batch_tgt = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_tgt_len)[tgt_valid]
    flat_tgt_values = target_ids[tgt_valid]

    labels[flat_batch_tgt, flat_dest_tgt] = flat_tgt_values
    # input_embeds:  [ audio frames ] [ prompt tokens ] [ padding if any ]
    # labels:        [ -100         ] [ -100          ] [ target tokens  ]

    return out_embs, labels


# ============================================================
# Trainer Builder
# ============================================================
def build_model_and_trainer(
    llm_path,
    audio_path,
    project_path,
    chunk_size,
    stride,
    stack_size,
    rank_dim,
    train,
    eval,
    max_steps,
    batch_size,
    bucket_size,
    max_seq_len,
    lr,
    weight_decay,
    eval_steps,
    logging_steps,
    output_dir,
):
    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")
    half_precision = (dtype==torch.bfloat16)

    ### 1. Load models    
    ### ============================================================

    # AudioEmbedder (frozen)
    audio_embedder = AudioEmbedder(
        model=audio_path, 
        l2_norm=False, 
        half_precision=half_precision, 
        chunk_size=chunk_size, 
        stride=stride, 
        device=args.device)

    audio_embedder.eval()
    for p in audio_embedder.parameters():
        p.requires_grad = False

    # LLM (frozen)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=dtype, device_map="auto")
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    # Projector (trainable)
    projector = AudioToLLMProjector(
        audio_embedding_dim=audio_embedder.config.hidden_size, 
        stack_size=stack_size, 
        llm_dimension=llm.config.hidden_size,
        rank_dim=rank_dim, 
        max_seq_len=max_seq_len)

    if project_path is not None:
        projector.load(project_path, device=device)

    ### 2. Datasets
    ### ============================================================

    train_dataset = AudioDatasetIterable(path=train, split="train")
    eval_dataset  = AudioDataset(path=eval, split="eval")
        

    def build_prompt(sample):
        has_src = bool(sample.get("lang"))
        has_tgt = bool(sample.get("tgt_lang"))

        if has_src and has_tgt:
            return f"\nTranscribe then translate into {sample['tgt_lang']}.\n[ASR]"
        elif has_src:
            return f"\nTranscribe.\n[ASR]"
        elif has_tgt:
            return f"\nTranslate into {sample['tgt_lang']}.\n[STT]"
        else:
            raise ValueError("Either 'lang' or 'tgt_lang' must be provided in the dataset.")


    def preprocess_fn(batch):
        """
        Expects `batch` to be a list-like batch from the datasets library where
        batch["audio"] and batch["target"] are lists of length B (or correspondingly shaped).
        Returns batched `input_embeds` [B, L_in, D] and `labels` [B, L_in] (with -100 in ignored positions).
        """
        audios = batch["audio"] # [B, 1, T]
        target_texts = batch["target"] # target reference texts
        B = len(audios)

        # Audio embeddings
        #####################
        # wavtok (frozen) - encode audio to embeddings
        with torch.no_grad():
            wav_list = wavtok.encode(audios) # list of B tensors [frame_i_lengh, audio_dim]
        audio_lens = torch.tensor([w.shape[0] for w in wav_list], device=device)  # [B]
        max_frames = audio_lens.max().item()
        # Create audio mask
        audio_mask = (torch.arange(max_frames, device=device).unsqueeze(0) < audio_lens.unsqueeze(1)).long()
        # stack with right-padded
        wav_embs = torch.nn.utils.rnn.pad_sequence(wav_list, batch_first=True, padding_value=0.0).to(device=device, dtype=dtype)     # [B, max_frames, audio_dim]
        # Projector - audio embeddings → superframes → projected embeddings
        proj_embs = projector(wav_embs).to(device=device, dtype=dtype) # [B, max_frames, llm_dim] (right-padded)

        # Prompt embeddings
        #####################
        # Build prompts with <extra_id_0> at beginning
        prompt_texts = [build_prompt(sample) for sample in batch]
        # Tokenize prompts and targets
        prompt_ids = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=False).input_ids.to(device)[:,1:].contiguous()  # [B, T_max_prompt-1] remove first token (<extra_id_0>)
        target_ids = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=False).input_ids.to(device)        # [B, L_max_target]
        # token embeddings from LLM embedding layer with right-padded
        with torch.no_grad():
            text_embs = llm.get_input_embeddings()(prompt_ids).to(device=device, dtype=dtype)  # [B, T_max_prompt-1, llm_dim] (right-padded)

        input_embeds, labels = compose_full_embeddings_with_padding_vectorized(
            proj_embs=proj_embs,
            audio_mask=audio_mask,
            text_embs=text_embs,
            prompt_ids=prompt_ids,
            target_ids=target_ids,
            device_local=device,
            dtype_local=dtype,
            max_seq_len=max_seq_len,
            pad_token_id=tokenizer.pad_token_id,
            ignore_index=-100,
        )
        return {
            "input_embeds": input_embeds,  # [B, L_in, D]
            "labels": labels,              # [B, L_in] with -100 for ignored positions
        }



    ### 3. SFTTrainer
    ### ============================================================

    trainer = SFTTrainer(
        model=llm,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="target",
        preprocessor=preprocess_fn,
        max_seq_length=max_seq_len,
        max_steps=max_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        bucket_size=bucket_size,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
    )

    return trainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a speech ASR/STT decoder (audio-embedder ➔ Projector ➔ LLM).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--llm_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    parser.add_argument("--audio_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147")
    parser.add_argument("--project_path", type=str, default=None)

    parser.add_argument("--chunk_size", type=int, default=3200, help="Group this many samples when building chunks in audio processor")
    parser.add_argument("--stride", type=int, default=1600, help="Overlap this many samples when building chunks in audio processor")
    parser.add_argument("--stack_size", type=int, default=16, help="Stack this many frames in audio to LLM projector")
    parser.add_argument("--rank_dim", type=int, default=256, help="Rank dimension for audio to LLM projector")

    parser.add_argument("--train", required=True, help="Training dataset file")
    parser.add_argument("--eval", required=True, help="Evaluation dataset file")

    parser.add_argument("--max_steps", type=int, default=50000, help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--bucket_size", type=int, default=32768, help="Bucket size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    parser.add_argument("--eval_steps", type=int, default=1000, help="Run evaluation after this many steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging after this many steps")

    parser.add_argument("--output_dir", type=str, default="./sft_output", help="Output directory of training")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    # Log file name with timestamp
    log_filename = os.path.join(args.output_dir, f"train.log") #_{datetime.now().strftime('%Y%m%d_%H%M%S')}
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),  # save to file
            #logging.StreamHandler()  # and print to console
        ]
    )

    logger = logging.getLogger(__name__)
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
    logger.info(f"Git commit: {get_git_commit()}")

    trainer = build_model_and_trainer(        
        llm_path=args.llm_path,
        audio_path=args.audio_path,
        project_path=args.project_path,
        chunk_size = args.chunk_size,
        stride=args.stride,
        stack_size=args.stack_size,
        rank_dim=args.rank_dim,
        train=args.train,
        eval=args.eval,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        bucket_size=args.bucket_size,
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
    )

    trainer.train()
