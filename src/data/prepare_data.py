import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio

def load_vivos_dataset(base_path):
    all_samples = []

    for split in ["train", "test"]:
        wav_dir = os.path.join(base_path, split, "waves")
        prompt_file = os.path.join(base_path, split, "prompts.txt")

        # Phân tách bằng regex khoảng trắng
        df = pd.read_csv(prompt_file, sep=r"\s+", engine="python", header=None, names=["file_id", "transcription"])
        
        for _, row in df.iterrows():
            file_id = row["file_id"]
            text = row["transcription"]
            speaker = file_id.split("_")[0]
            wav_path = os.path.join(wav_dir, speaker, file_id + ".wav")
            if os.path.exists(wav_path):
                all_samples.append({"audio": wav_path, "transcription": text})

    return DatasetDict({"train": Dataset.from_list(all_samples)})

def preprocess_dataset(dataset_dict):
    dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=16000))
    return dataset_dict
