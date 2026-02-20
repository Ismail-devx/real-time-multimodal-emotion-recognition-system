import os
import numpy as np
import torch
import pickle
from datasets import Dataset, DatasetDict
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

# Set Hugging Face and WandB tokens
os.environ["HF_TOKEN"] = "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"
wandb.login(key="0b96622bd8f68f910b8d560246575a5b8a2ff898")

# Define processed data paths
processed_data_paths = {
    'audio': "C:/Users/ismai/PycharmProjects/EmotionRecognitionSystem/Datasets/processed_audio_data.pkl"
}

# Load preprocessed audio data
def load_pickle_data(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None

audio_data = load_pickle_data(processed_data_paths['audio'])

if audio_data is None:
    print("Error: Audio dataset could not be loaded. Please check the file path.")
else:
    # Merge speech datasets
    X_speech = audio_data['features']
    y_speech = audio_data['labels']

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_speech, y_speech, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert speech data to Hugging Face Dataset
    speech_dataset = DatasetDict({
        "train": Dataset.from_dict({"input_values": X_train.tolist(), "labels": y_train.tolist()}),
        "validation": Dataset.from_dict({"input_values": X_val.tolist(), "labels": y_val.tolist()}),
        "test": Dataset.from_dict({"input_values": X_test.tolist(), "labels": y_test.tolist()})
    })

    # Load pre-trained Wav2Vec2 and processor with error handling
    wav2vec_model_name = "facebook/wav2vec2-large-960h-lv60-self"
    try:
        print("Loading pre-trained Wav2Vec2 model and processor...")
        wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            wav2vec_model_name,
            num_labels=len(set(y_speech)),
            use_auth_token=True
        )
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            wav2vec_model_name,
            use_auth_token=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If this persists, manually download the model from Hugging Face and place it in the correct directory.")
        exit()

    # Preprocess speech data
    def preprocess_speech_function(examples):
        return wav2vec_processor(examples["input_values"], sampling_rate=16000, return_tensors="pt", padding=True)

    speech_dataset = speech_dataset.map(preprocess_speech_function, batched=True)

    # Training arguments for Wav2Vec2
    wav2vec_training_args = TrainingArguments(
        output_dir="C:/Users/ismai/PycharmProjects/EmotionRecognitionSystem/Models/wav2vec2_finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        fp16=True,
        gradient_accumulation_steps=2,
        report_to="wandb",
        logging_steps=10,
        log_level="info"
    )

    # Trainer for Wav2Vec2
    wav2vec_trainer = Trainer(
        model=wav2vec_model,
        args=wav2vec_training_args,
        train_dataset=speech_dataset["train"],
        eval_dataset=speech_dataset["validation"]
    )

    # Train Wav2Vec2
    print("Starting training for Wav2Vec2...")
    wav2vec_trainer.train()
    print("Wav2Vec2 training completed.")

    # Save fine-tuned model
    wav2vec_model.save_pretrained("C:/Users/ismai/PycharmProjects/EmotionRecognitionSystem/Models/wav2vec2_finetuned")
    wav2vec_processor.save_pretrained("C:/Users/ismai/PycharmProjects/EmotionRecognitionSystem/Models/wav2vec2_finetuned")

#Loaded a pre-trained Wav2Vec2 model from huggingface
#Loaded your audio data and matched it with emotion labels.
#Tokenized the audio using Wav2Vec’s processor.
#Trained the model using CrossEntropy loss and saved it locally.
#Likely used Trainer, torch.nn.Module, or a custom loop.