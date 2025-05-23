import os
import torchaudio
from datasets import Dataset, DatasetDict, Audio

def load_vivos_dataset(base_path):
    data = {'train': [], 'test': []}
    for split in ['test']:
        audio_dir = os.path.join(base_path, split, 'wav')
        label_path = os.path.join(base_path, split, 'txt')
        for speaker in os.listdir(audio_dir):
            speaker_dir = os.path.join(audio_dir, speaker)
            for wav_file in os.listdir(speaker_dir):
                audio_path = os.path.join(speaker_dir, wav_file)
                transcript_file = os.path.join(label_path, speaker, wav_file.replace(".wav", ".txt"))
                if os.path.exists(transcript_file):
                    with open(transcript_file, 'r', encoding='utf8') as f:
                        transcript = f.read().strip()
                    data[split].append({
                        "audio": audio_path,
                        "transcription": transcript
                    })
    return DatasetDict({
        "test": Dataset.from_list(data["test"])
    })

def preprocess_dataset(dataset_dict):
    dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=16000))
    return dataset_dict
