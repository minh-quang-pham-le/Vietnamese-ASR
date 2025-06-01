import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModelForSeq2SeqLM
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import kenlm
import json
from tqdm import tqdm
import re
from huggingface_hub import hf_hub_download

# 1. Load mô hình
MODEL_DIR = "wav2vec2_vi_ft"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR, cache_dir="./cache").to(device).eval()
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR, cache_dir="./cache")
print(processor.tokenizer.get_vocab())

# Language model path
lm_path = hf_hub_download(
    repo_id="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    filename="vi_lm_4grams.bin.zip",
    cache_dir="./cache"
)

kenlm_path = "./cache/vi_lm_4grams.bin"

# Giải nén
import zipfile
with zipfile.ZipFile(lm_path, 'r') as zip_ref:
    zip_ref.extractall("./cache")

# 2. Load mô hình sửa lỗi chính tả 
corr_tokenizer = AutoTokenizer.from_pretrained("bmd1905/vietnamese-correction")
corr_model = AutoModelForSeq2SeqLM.from_pretrained("bmd1905/vietnamese-correction")

def remove_punctuation(text):
    return re.sub(r"[^\w\sÀ-ỹ]", "", text)

def correct_text(text):
    inputs = corr_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    corr_model.to(device)

    with torch.no_grad():
        outputs = corr_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128
        )
    corrected = corr_tokenizer.decode(outputs[0], skip_special_tokens=True)
    corrected = remove_punctuation(corrected)
    return corrected.lower().strip()

# 3. Load vocab
vocab_dict = processor.tokenizer.get_vocab()
sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
vocab = [x[1] for x in sort_vocab][:-2]
vocab_list = vocab
# Convert ctc blank character representation
vocab_list[processor.tokenizer.pad_token_id] = ""
# Replace special characters
vocab_list[processor.tokenizer.unk_token_id] = ""
# vocab_list[tokenizer.bos_token_id] = ""
# vocab_list[tokenizer.eos_token_id] = ""
# Convert space character representation
vocab_list[processor.tokenizer.word_delimiter_token_id] = " "
print(f"Vocab size: {len(vocab_list)}")

# 4. Tạo decoder dùng KenLM
alphabet = Alphabet(vocab_list, is_bpe=False)
kenlm_model = kenlm.Model(kenlm_path)
decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(kenlm_model))

# 5. Load danh sách test
PROMPT_CSV = "prompts_asr.csv"
AUDIO_DIR = "data/test/private-test-data-asr"

df = pd.read_csv(PROMPT_CSV)
audio_files = df["path"].tolist()

# 6. Hàm nhận dạng dùng KenLM + beam search
def transcribe_beam(audio_path):
    speech_array, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech_array = resampler(speech_array)

    inputs = processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits[0].cpu().detach().numpy()  # Shape (T, V)
    
    # Beam Search decode using KenLM
    transcription = decoder.decode(logits, beam_width=500)
    clean_text = " ".join(transcription.lower().strip().split())  # Chuẩn hóa khoảng trắng
    final_text = correct_text(clean_text)
    return final_text

# 7. Ghi kết quả vào transcripts.txt
with open("transcripts.txt", "w", encoding="utf-8") as out_f:
    for fname in tqdm(audio_files):
        path = os.path.join(AUDIO_DIR, fname)
        transcript = transcribe_beam(path)
        out_f.write(transcript + "\n")
        print(f"[Done]{fname} → {transcript}")