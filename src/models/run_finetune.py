import torch
import yaml
from transformers import AutoProcessor, AutoModelForCTC, TrainingArguments, Trainer, DataCollatorCTCTokenizer
from data.prepare_data import load_vivos_dataset, preprocess_dataset
from utils.metrics import compute_wer
from datasets import load_metric

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Load processor and model
processor = AutoProcessor.from_pretrained(config["model_checkpoint"])
model = AutoModelForCTC.from_pretrained(config["model_checkpoint"])

# Load and preprocess dataset
dataset = load_vivos_dataset(config["dataset_path"])
dataset = preprocess_dataset(dataset)

# Preprocessing
def prepare_batch(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=16000, return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels = processor(batch["transcription"], return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]
    batch["labels"] = labels.input_ids[0]
    return batch

dataset = dataset.map(prepare_batch, remove_columns=dataset["test"].column_names)

# Training
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    per_device_train_batch_size=config["batch_size"],
    evaluation_strategy="steps",
    eval_steps=config["eval_steps"],
    logging_steps=50,
    num_train_epochs=config["num_train_epochs"],
    learning_rate=config["learning_rate"],
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    report_to="none",
)

data_collator = DataCollatorCTCTokenizer(processor=processor, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,  # Không dùng train vì chỉ finetune test
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_wer
)

trainer.evaluate()