import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset, DatasetDict, Audio
from jiwer import wer, Compose, ToLowerCase, RemoveMultipleSpaces
        
if __name__ == "__main__":
    # --- Load ground-truth transcripts ---
    transcript_dict = {}
    with open("data/train/vivos/test/prompts.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                audio_id, text = parts
                transcript_dict[audio_id] = text.lower().strip()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model & processor ---
    model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-vivos-ft").to(device)
    processor = Wav2Vec2Processor.from_pretrained("wav2vec2-vivos-ft")
    model.eval()

    # --- Duyệt file audio ---
    WAV_DIR = "data/train/vivos/test/waves"
    preds = []
    refs = []

    print(f"{'ID':<20} | {'Ground truth':<50} | Prediction")
    print("-" * 100)

    for root, _, files in os.walk(WAV_DIR):
        for fname in files:
            if not fname.endswith(".wav"):
                continue

            audio_id = os.path.splitext(fname)[0]
            if audio_id not in transcript_dict:
                continue

            path = os.path.join(root, fname)

            # Load audio
            try:
                speech_array, sr = torchaudio.load(path)
            except Exception as e:
                print(f"[!] Lỗi đọc {path}: {e}")
                continue

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                speech_array = resampler(speech_array)

            # Predict
            input_values = processor(
                speech_array.squeeze(), sampling_rate=16000, return_tensors="pt"
            ).input_values.to(device)

            with torch.no_grad():
                logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            print("Raw pred_ids:", pred_ids[0].tolist())
            transcription = processor.batch_decode(pred_ids)[0].lower().strip()

            # Ground truth
            ground_truth = transcript_dict[audio_id]

            if transcription and ground_truth:
                preds.append(transcription)
                refs.append(ground_truth)
                print(f"{audio_id:<20} | {ground_truth:<50} | {transcription}")

    # --- Tính WER ---
    print("\n" + "-" * 100)
    if len(refs) == 0:
        print("Không có mẫu hợp lệ nào để đánh giá WER.")
    else:
        transform = Compose([ToLowerCase(), RemoveMultipleSpaces()])
        score = wer(refs, preds, truth_transform=transform, hypothesis_transform=transform)
        print(f"WER trên tập test VIVOS: {score * 100:.2f}%")