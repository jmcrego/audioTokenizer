import os
import torch
import logging
import argparse
import subprocess
import numpy as np
from trl import SFTTrainer, SFTConfig
#from torch.utils.data import Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from AudioEmbedder import AudioEmbedder
from AudioToLLMProjector import AudioToLLMProjector
from Datasets import build_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return commit
    except Exception:
        return "unknown"

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


# class BucketedLengthSampler(Sampler):
#     """
#     Buckets dataset by total_length, shuffles within each bucket, and yields batches.
#     """
#     def __init__(self, dataset, batch_size, bucket_size=1000, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.bucket_size = bucket_size
#         self.shuffle = shuffle

#         # extract total_length from dataset
#         self.lengths = np.array([s["total_length"] for s in dataset])
#         self.sorted_indices = np.argsort(self.lengths)

#     def __iter__(self):
#         # split sorted indices into buckets
#         buckets = [
#             self.sorted_indices[i:i+self.bucket_size]
#             for i in range(0, len(self.sorted_indices), self.bucket_size)
#         ]

#         all_indices = []
#         for b in buckets:
#             if self.shuffle:
#                 b = np.random.permutation(b)
#             all_indices.extend(b.tolist())

#         # yield batches
#         for i in range(0, len(all_indices), self.batch_size):
#             yield all_indices[i:i+self.batch_size]

#     def __len__(self):
#         return len(self.dataset)
    
# ============================================================
# Compose batch of embeddings/targets with right-padding (vectorized)
# ============================================================
def compose_full_embeddings_with_padding_vectorized(
    proj_embs: torch.Tensor,  # [B, S, D] (audio embeddings, right-padded)
    audio_mask: torch.Tensor, # [B, S]    (1=real, 0=pad)
    prompt_embs: torch.Tensor,# [B, T, D] (prompt embeddings, right-padded)
    prompt_ids: torch.Tensor, # [B, T]    (for detecting padding)
    target_ids: torch.Tensor, # [B, L]    (padded with pad_token_id)
    device,
    dtype,
    max_seq_len: int,
    pad_token_id: int,
    ignore_index: int = -100,
):
    """
    Concatenate audio embeddings + prompt embeddings right-padded
    and return final padded input_embeds and labels.

    For instance, 
    Input EMBEDDINGS should be:
    [ a a a a p p p 0 0 0 0 0 0] 
    [ a a p p 0 0 0 0 0 0 0 0 0] 
    [ a a a p p p 0 0 0 0 0 0 0]
    (a means audio embedding, p means prompt embedding 0 means pad embedding)

    While LABELS should be:
    [ -100 -100 -100 -100 -100 -100 -100 -100 t t t    t -100]
    [ -100 -100 -100 -100 -100 -100 -100 -100 t t t -100 -100]
    [ -100 -100 -100 -100 -100 -100 -100 -100 t t t    t    t]
    (t means label token)

    Returns:
        input_embeds: [B, L_final, D]
        labels:       [B, L_final]
    """

    B, S, D = proj_embs.shape
    _, T, _ = prompt_embs.shape
    _, L = target_ids.shape

    # Determine real lengths
    audio_lens = audio_mask.sum(dim=1)                    # [B]
    prompt_lens = (prompt_ids != pad_token_id).sum(dim=1) # [B]
    target_lens = (target_ids != pad_token_id).sum(dim=1) # [B]

    # Total length per sequence
    total_lens = audio_lens + prompt_lens + target_lens
    max_len = min(total_lens.max().item(), max_seq_len)

    # -----------------------
    # Input embeddings
    # -----------------------
    input_embeds = torch.zeros((B, max_len, D), device=device, dtype=dtype)

    # Audio embeddings
    audio_idx = torch.arange(S, device=device).view(1, -1).expand(B, -1)  # [B, S]
    audio_valid = audio_idx < audio_lens.unsqueeze(1)                     # [B, S]
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, S) # [B, S]
    input_embeds[batch_idx[audio_valid], audio_idx[audio_valid]] = proj_embs[audio_valid]

    # Prompt embeddings
    prompt_idx = torch.arange(T, device=device).view(1, -1).expand(B, -1) # [B, T]
    prompt_valid = prompt_idx < prompt_lens.unsqueeze(1)
    dest_positions = audio_lens.unsqueeze(1) + prompt_idx                 # [B, T]
    dest_positions = torch.clamp(dest_positions, max=max_len-1)
    batch_idx_prompt = torch.arange(B, device=device).unsqueeze(1).expand(-1, T)
    input_embeds[batch_idx_prompt[prompt_valid], dest_positions[prompt_valid]] = prompt_embs[prompt_valid]

    # -------------------
    # Labels
    # -------------------
    labels = torch.full((B, max_len), ignore_index, dtype=torch.long, device=device)

    target_idx = torch.arange(L, device=device).view(1, -1).expand(B, -1)
    target_valid = target_idx < target_lens.unsqueeze(1)
    dest_target_positions = audio_lens.unsqueeze(1) + prompt_lens.unsqueeze(1) + target_idx
    dest_target_positions = torch.clamp(dest_target_positions, max=max_len-1)
    batch_idx_target = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)
    labels[batch_idx_target[target_valid], dest_target_positions[target_valid]] = target_ids[target_valid]

    return input_embeds, labels

# ============================================================
# Trainer Builder
# ============================================================
def build_model_and_trainer(
    llm,
    audio,
    proj,
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
    eval_steps,
    logging_steps,
    output_dir,
):
    device, dtype = get_device_dtype()
    logger.info(f"device: {device}, dtype: {dtype}")

    ### 1. Load models    
    ### ============================================================

    # AudioEmbedder (frozen)
    audio_embedder = AudioEmbedder(
        model=audio, 
        l2_norm=False, 
        chunk_size=chunk_size, 
        stride=stride, 
        device=device,
        dtype=dtype)

    audio_embedder.eval()
    for p in audio_embedder.parameters():
        p.requires_grad = False

    # Tokenizer + LLM (frozen)
    tokenizer = AutoTokenizer.from_pretrained(llm, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(llm, dtype=dtype, device_map="auto")
    llm_model.eval()
    for p in llm_model.parameters():
        p.requires_grad = False

    # Projector (trainable)
    projector = AudioToLLMProjector(
        audio_embedding_dim=audio_embedder.D, 
        stack_size=stack_size, 
        llm_dimension=llm_model.config.hidden_size,
        rank_dim=rank_dim, 
        max_seq_len=max_seq_len)

    # load if given path
    if proj is not None:
        projector.load(proj, device=device)

    ### 2. Datasets
    ### ============================================================

    asr_token = "[ASR]"
    stt_token = "[STT]"
    end_token = "[END]"
    sample_rate = audio_embedder.sample_rate if hasattr(audio_embedder, "sample_rate") else 16000

    # train_dataset = AudioDataset(path=train, tokenizer=tokenizer, 
    #                              asr_token=asr_token, stt_token=stt_token, end_token=end_token,
    #                              sample_rate=sample_rate, chunk_size=chunk_size, stride=stride, stack_size=stack_size, max_seq_len=max_seq_len)    
    # train_sampler = BucketedLengthSampler(train_dataset, batch_size=batch_size, bucket_size=bucket_size)
    # train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=lambda batch: batch)

    # eval_dataset  = AudioDataset(path=eval, tokenizer=tokenizer, 
    #                              asr_token=asr_token, stt_token=stt_token, end_token=end_token,
    #                              sample_rate=sample_rate, chunk_size=chunk_size, stride=stride, stack_size=stack_size, max_seq_len=max_seq_len)
    # eval_sampler = BucketedLengthSampler(eval_dataset, batch_size=batch_size, bucket_size=bucket_size, shuffle=False)
    # eval_loader = DataLoader(eval_dataset, batch_sampler=eval_sampler, collate_fn=lambda batch: batch) 

    train_dataset = build_dataset(
        file_path=train,
        tokenizer=tokenizer,
        asr_token=asr_token,
        stt_token=stt_token,
        end_token=end_token,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        stride=stride,
        stack_size=stack_size,
        max_seq_len=max_seq_len
    )

    eval_dataset = build_dataset(
        file_path=eval,
        tokenizer=tokenizer,
        asr_token=asr_token,
        stt_token=stt_token,
        end_token=end_token,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        stride=stride,
        stack_size=stack_size,
        max_seq_len=max_seq_len
    )

    train_dataset = train_dataset.with_format("python")
    eval_dataset = eval_dataset.with_format("python")

    def preprocess_fn(batch):
        """
        Expects `batch` to be a list-like batch from the datasets library where
        batch["audio"] and batch["target"] are lists of length B (or correspondingly shaped).
        Returns batched `input_embeds` [B, L_in, D] and `labels` [B, L_in] (with -100 in ignored positions).
        """

        device_local, dtype_local = device, dtype

        # batch is dict of lists
        audios = batch["audio_path"]

        prompt_list = [
            torch.tensor(ids, dtype=torch.long)
            for ids in batch["prompt_ids"]
        ]
        target_list = [
            torch.tensor(ids, dtype=torch.long)
            for ids in batch["target_ids"]
        ]

        # Padding
        pad_id = tokenizer.pad_token_id
        prompt_ids = pad_sequence(prompt_list, batch_first=True, padding_value=pad_id).to(device_local)
        target_ids = pad_sequence(target_list, batch_first=True, padding_value=pad_id).to(device_local)

        # Audio embedder
        with torch.no_grad():
            embs, embs_mask = audio_embedder(audios)
            if embs.dtype != dtype_local:
                embs = embs.to(dtype=dtype_local)
            embs_mask = embs_mask.bool()

        # Projection
        proj_embs = projector(embs)

        # Prompt embedding lookup
        with torch.no_grad():
            prompt_embs = llm_model.get_input_embeddings()(prompt_ids)
            if prompt_embs.dtype != dtype_local:
                prompt_embs = prompt_embs.to(dtype=dtype_local)

        # Compose multimodal sequence
        input_embeds, labels = compose_full_embeddings_with_padding_vectorized(
            proj_embs=proj_embs,
            audio_mask=embs_mask,
            prompt_embs=prompt_embs,
            prompt_ids=prompt_ids,
            target_ids=target_ids,
            device=device_local,
            dtype=dtype_local,
            max_seq_len=max_seq_len,
            pad_token_id=pad_id,
            ignore_index=-100,
        )

        return {
            "input_embeds": input_embeds,
            "labels": labels,
        }

        # device_local, dtype_local = device, dtype  # capture outer scope

        # # batch is a list of dicts
        # audios = [sample["audio_path"] for sample in batch]

        # prompt_list = [
        #     torch.tensor(sample["prompt_ids"], dtype=torch.long)
        #     for sample in batch
        # ]
        # target_list = [
        #     torch.tensor(sample["target_ids"], dtype=torch.long)
        #     for sample in batch
        # ]

        # pad_id = tokenizer.pad_token_id

        # prompt_ids = pad_sequence(prompt_list, batch_first=True, padding_value=pad_id).to(device_local)
        # target_ids = pad_sequence(target_list, batch_first=True, padding_value=pad_id).to(device_local)

        # # Encode audio
        # with torch.no_grad():
        #     embs, embs_mask = audio_embedder(audios)
        #     if embs.dtype != dtype_local:
        #         embs = embs.to(dtype=dtype_local)
        #     embs_mask = embs_mask.bool()

        # # Project
        # proj_embs = projector(embs)

        # # Prompt embeddings
        # with torch.no_grad():
        #     prompt_embs = llm_model.get_input_embeddings()(prompt_ids)
        #     if prompt_embs.dtype != dtype_local:
        #         prompt_embs = prompt_embs.to(dtype=dtype_local)

        # # Compose full sequence
        # input_embeds, labels = compose_full_embeddings_with_padding_vectorized(
        #     proj_embs=proj_embs,
        #     audio_mask=embs_mask,
        #     prompt_embs=prompt_embs,
        #     prompt_ids=prompt_ids,
        #     target_ids=target_ids,
        #     device=device_local,
        #     dtype=dtype_local,
        #     max_seq_len=max_seq_len,
        #     pad_token_id=pad_id,
        #     ignore_index=-100,
        # )

        # return {
        #     "input_embeds": input_embeds,
        #     "labels": labels,
        # }

        # # Extract batch fields
        # audios = batch["audio_path"]

        # prompt_list = [
        #     torch.tensor(x, dtype=torch.long)
        #     for x in batch["prompt_ids"]
        # ]
        # target_list = [
        #     torch.tensor(x, dtype=torch.long)
        #     for x in batch["target_ids"]
        # ]

        # pad_id = tokenizer.pad_token_id

        # # Stack prompt and target sequences directly on GPU
        # prompt_ids = pad_sequence(prompt_list, batch_first=True, padding_value=pad_id).to(device_local)
        # target_ids = pad_sequence(target_list, batch_first=True, padding_value=pad_id).to(device_local)

        # # Encode audio to embeddings on GPU
        # with torch.no_grad():
        #     embs, embs_mask = audio_embedder(audios)  # [B, T, D], [B, T] : B batch size, T frame lengh, D audio dim (right-padded)
        #     # Only move to dtype if needed, assume embedder outputs on correct device
        #     if embs.dtype != dtype_local:
        #         embs = embs.to(dtype=dtype_local)
        #     embs_mask = embs_mask.bool()  # ensure mask is boolean

        # # Project audio embeddings to LLM dimension
        # proj_embs = projector(embs) # [B, N3, D2] (right-padded)

        # # Get token embeddings for prompt
        # with torch.no_grad():
        #     prompt_embs = llm_model.get_input_embeddings()(prompt_ids) # [B, T_max_prompt-1, llm_dim] (right-padded)
        #     if prompt_embs.dtype != dtype_local:
        #         prompt_embs = prompt_embs.to(dtype=dtype_local)

        # # Compose final input embeddings and labels
        # input_embeds, labels = compose_full_embeddings_with_padding_vectorized(
        #     proj_embs=proj_embs,
        #     audio_mask=embs_mask,
        #     prompt_embs=prompt_embs,
        #     prompt_ids=prompt_ids,
        #     target_ids=target_ids,
        #     device=device_local,
        #     dtype=dtype_local,
        #     max_seq_len=max_seq_len,
        #     pad_token_id=pad_id,
        #     ignore_index=-100,
        # )

        # return {
        #     "input_embeds": input_embeds,  # [B, L_in, D]
        #     "labels": labels,              # [B, L_in] with -100 for ignored positions
        # }

    ### 3. SFTTrainer
    ### ============================================================
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
    )

    # def data_collator(batch):
    #     return preprocess_fn(batch)

    trainer = SFTTrainer(
        model=llm_model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=preprocess_fn,
        processing_class=tokenizer,
    )

    return trainer



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a speech ASR/STT decoder (audio-embedder ➔ Projector ➔ LLM).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model paths
    parser.add_argument("--llm", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B-Instruct")
    parser.add_argument("--audio", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/mHuBERT-147")
    parser.add_argument("--proj", type=str, default=None)
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
    parser.add_argument("--bucket_size", type=int, default=1000, help="Bucket size")
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
        llm=args.llm,
        audio=args.audio,
        proj=args.proj,
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
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
    )

    trainer.train()
