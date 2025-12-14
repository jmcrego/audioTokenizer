
import torch
import logging

logger = logging.getLogger("AudioToLLMGeneratorHF")

class AudioToLLMGeneratorHF:
    def __init__(
        self,
        model,
        tokenizer,
        audio_embedder,
        projector,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.audio_embedder = audio_embedder
        self.projector = projector

        # infer device and dtype from model
        param = next(model.parameters())
        self.device = param.device
        self.dtype  = param.dtype
         
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.model.eval()

    @torch.no_grad()
    def generate(self, audio_files, prompt: str):
        """
        audio_files: list[str]
        prompt: text instruction ("transcribe and translate", etc.)
        """

        # --------------------------------------------------
        # 1) Audio → embeddings → LLM projected embeddings 
        # --------------------------------------------------
        audio_embs, audio_mask = self.audio_embedder(audio_files)
        audio_embs = audio_embs.to(self.device, self.dtype)
        audio_mask = audio_mask.to(self.device)

        proj_embs, proj_mask = self.projector(audio_embs, audio_mask)
        proj_embs = proj_embs.to(self.device, self.dtype)
        proj_mask = proj_mask.to(self.device)

        B, S, D = proj_embs.shape

        # --------------------------------------------------
        # 2) Prompt embeddings
        # --------------------------------------------------
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)

        prompt_embs = self.model.get_input_embeddings()(prompt_ids)
        prompt_embs = prompt_embs.expand(B, -1, -1)

        # --------------------------------------------------
        # 3) Concatenate embeddings
        # --------------------------------------------------
        inputs_embeds = torch.cat([proj_embs, prompt_embs], dim=1)

        attention_mask = torch.cat(
            [
                proj_mask,
                torch.ones(
                    (B, prompt_embs.size(1)),
                    device=self.device,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )

        # --------------------------------------------------
        # 4) Generate
        # --------------------------------------------------
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # --------------------------------------------------
        # 6) Decode ONLY generated tokens
        # --------------------------------------------------
        gen_tokens = outputs[:, inputs_embeds.size(1):]
        texts = self.tokenizer.batch_decode(
            gen_tokens,
            skip_special_tokens=True,
        )

        return texts
