import os
from datasets import Dataset, DatasetDict, Audio

def load_vivos_dataset(base_path):
    all_samples = []

    for split in ["train", "test"]:
        wav_dir = os.path.join(base_path, split, "waves")
        prompt_file = os.path.join(base_path, split, "prompts.txt")

        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                file_id, text = line.split(" ", 1)
                speaker = file_id.split("_")[0]
                wav_path = os.path.join(wav_dir, speaker, file_id + ".wav")
                if os.path.exists(wav_path):
                    all_samples.append({"audio": wav_path, "transcription": text})

    return DatasetDict({"train": Dataset.from_list(all_samples)})

def preprocess_dataset(dataset_dict):
    dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=16000))
    return dataset_dict

if __name__ == "__main__":
    base_path = "data\\train\\vivos"
    dataset = load_vivos_dataset(base_path)
    dataset = preprocess_dataset(dataset)
    print(dataset)