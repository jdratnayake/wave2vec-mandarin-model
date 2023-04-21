from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

api_token = "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"
model_name = "kehanlu/mandarin-wav2vec2-aishell1"
cache_dir = "model_cache"

class ExtendedWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """
    In ESPNET there is a LayerNorm layer between encoder output and CTC classification head.
    """
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = torch.nn.Sequential(
                torch.nn.LayerNorm(config.hidden_size),
                self.lm_head
        )
        
model = ExtendedWav2Vec2ForCTC.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1", use_auth_token=api_token, cache_dir=cache_dir)
processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1", use_auth_token=api_token, cache_dir=cache_dir)