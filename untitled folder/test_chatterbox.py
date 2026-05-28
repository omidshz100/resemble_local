import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Use "mps" for Apple Silicon (M1/M2/M3), "cpu" for Intel
device = "mps"

print(f"Loading model on {device}...")
model = ChatterboxTTS.from_pretrained(device=device)

text = "Hello! Chatterbox TTS is working on your laptop."
print("Generating speech...")
wav = model.generate(text)

output_path = "output.wav"
# resample to 44100 Hz for universal compatibility
resampler = ta.transforms.Resample(orig_freq=model.sr, new_freq=44100)
wav_resampled = resampler(wav)
ta.save(output_path, wav_resampled, 44100)
print(f"Saved to {output_path} (44100 Hz)")
