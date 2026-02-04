
def build_template(
        type="oneline", 
        # asr: automatic speech recognition, 
        # ast: automatic speech translation, 
        # stt: speech transcription and translation, 
        # ttt: text to text translation
        task="asr", 
        audio_token="<extra_id_0>", 
        bos_token="<s>", #or <|start_audio|>
        eos_token="</s>", #or <|end_audio|>
        src_lang=None, 
        tgt_lang=None, 
        src_text=None, 
        tgt_text=None,
    ):

    prompt = None
    target = None
    
    if type not in {'instruct', 'projector', 'oneline'}:
        raise ValueError("unknown template type: use 'instruct' OR 'projector' OR 'oneline'")

    if task not in {'asr', 'ast', 'stt', 'ttt'}:
        raise ValueError("unknown template task: use 'asr' OR 'ast' OR 'stt' OR 'ttt'")

    if type == "oneline":

        # Automatic Speech Recognition
        if task == "asr":
            prompt=f"{audio_token}<|{task}|>" 
            target=f"<|{src_lang}|>{src_text}" if src_lang is not None and src_text is not None else None

        # Automatic Speech Translation
        elif task == "ast":
            if tgt_lang is None:
                return prompt, target
            
            prompt=f"{audio_token}<|{task}|><|{tgt_lang}|>" 
            target=f"<|{src_lang}|>{tgt_text}" if src_lang is not None and tgt_text is not None else None

        # Speech Transcription and Translation
        elif task == "stt":
            if src_lang is None or tgt_lang is None:
                return prompt, target
            if src_text is None:
                return prompt, target
            
            prompt=f"{audio_token}<|{task}-asr|><|{src_lang}|>{src_text}<|{task}|><|{tgt_lang}|>" 
            target=f"{tgt_text}" if tgt_text is not None else None

        # Text to Text Translation (No audio involved)
        elif task == "ttt":
            if tgt_lang is None:
                return prompt, target
            
            prompt=f"{src_text}<|{task}|><|{tgt_lang}|>"
            target=f"<|{src_lang}|>{tgt_text}" if src_lang is not None and tgt_text is not None else None

        return bos_token+prompt, target+eos_token if target is not None else None

    elif type == "projector":
        # ASR is not performed, only audio/text embeddings alignment, no inference required for this mode
        prompt=f"{audio_token}" 
        target=f"{bos_token}{src_text}{eos_token}"
        return prompt, target


    elif type == "instruct":
        raise NotImplementedError("instruct template not implemented yet")


