import torch
import soundfile as sf
import gradio as gr
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

api_token = "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"
model_name = "kehanlu/mandarin-wav2vec2-aishell1"
# model_name = "kehanlu/mandarin-wav2vec2"
cache_dir = "model_cache"

# sample rate
target_sr = 16000
# audio splitting duration (seconds)
time_units = 6

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

def convert(inputfile, outfile):
    data, sample_rate = librosa.load(inputfile)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)
    sf.write(outfile, data, target_sr)

def split_file(file_path):
    audio, samplerate = sf.read(file_path)
    segment_duration = time_units * samplerate
    
    count = 1
    for i in range(0, len(audio), segment_duration):
        segment = audio[i:i+segment_duration]
        sf.write(f"segment{count}.wav", segment, samplerate)
        count += 1
    
    return count - 1

model = ExtendedWav2Vec2ForCTC.from_pretrained(model_name, use_auth_token=api_token, cache_dir=cache_dir)
processor = Wav2Vec2Processor.from_pretrained(model_name, use_auth_token=api_token, cache_dir=cache_dir)

examples = [
    "Easy Mandarin 2 - What do you like about Taiwan.wav",
]

wav_file_path = examples[0]
filename = wav_file_path.split('.')[0]
new_file_path = filename + "16k.wav"
convert(wav_file_path, new_file_path)
audio_file_length = split_file(new_file_path)

transcription_text = ""
for i in range(1, audio_file_length + 1):
    audio_input, _ = sf.read(f"segment{i}.wav")
    inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt")

    with torch.no_grad():
        model.eval()
        logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
    transcription_text += transcription[0]

print(transcription_text)