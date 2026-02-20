import os
import torch
import pickle
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import wandb

# Set Hugging Face and WandB tokens
os.environ["HF_TOKEN"] = "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"
wandb.login(key="0b96622bd8f68f910b8d560246575a5b8a2ff898")

# Define processed text data path
processed_text_data_path = "C:/Users/ismai/PycharmProjects/EmotionRecognitionSystem/Datasets/processed_text_data.pkl"

# Load preprocessed text data
try:
    print("Loading preprocessed text data...")
    text_data = pd.read_pickle(processed_text_data_path)
    print("Text data loaded successfully.")
except FileNotFoundError:
    print(f"Error: {processed_text_data_path} not found.")
    exit()

# Convert label column to numeric class IDs
label_list = text_data["emotion"].unique().tolist()
label_to_id = {label: idx for idx, label in enumerate(label_list)}
text_data["label"] = text_data["emotion"].map(label_to_id)

# Load pre-trained BERT model and tokenizer
bert_model_name = "bert-base-uncased"
try:
    print("Loading BERT model and tokenizer...")
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, use_auth_token=True)
    bert_model = BertForSequenceClassification.from_pretrained(
        bert_model_name,
        num_labels=len(label_list),
        use_auth_token=True
    )
    print("BERT model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading BERT model: {e}")
    exit()

# Tokenization function for BERT
def tokenize_function(examples):
    return bert_tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Convert text data to Hugging Face Dataset
text_dataset = Dataset.from_pandas(text_data)

# Tokenize the dataset
print("Tokenizing text data...")
text_dataset = text_dataset.map(tokenize_function, batched=True)
text_dataset = text_dataset.rename_column("label", "labels")
text_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
print("Text data tokenized successfully.")

# Split text data into train, validation, and test sets
text_train_testvalid = text_dataset.train_test_split(test_size=0.4)
text_test_valid = text_train_testvalid['test'].train_test_split(test_size=0.5)

text_dataset = DatasetDict({
    'train': text_train_testvalid['train'],
    'validation': text_test_valid['train'],
    'test': text_test_valid['test']
})

# Training arguments for BERT
bert_training_args = TrainingArguments(
    output_dir="C:/Users/ismai/PycharmProjects/EmotionRecognitionSystem/Models/bert_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=True,
    gradient_accumulation_steps=2,
    report_to="wandb",
    logging_steps=10,
    log_level="info"
)

# Trainer for BERT
bert_trainer = Trainer(
    model=bert_model,
    args=bert_training_args,
    train_dataset=text_dataset["train"],
    eval_dataset=text_dataset["validation"]
)

# Train BERT
print("Starting BERT training...")
try:
    bert_trainer.train()
    print("BERT training completed.")
except Exception as e:
    print(f"Error during BERT training: {e}")

# Save fine-tuned BERT model
try:
    print("Saving fine-tuned BERT model...")
    bert_model.save_pretrained("C:/Users/ismai/PycharmProjects/EmotionRecognitionSystem/Models/bert_finetuned")
    bert_tokenizer.save_pretrained("C:/Users/ismai/PycharmProjects/EmotionRecognitionSystem/Models/bert_finetuned")
    print("BERT model saved successfully.")
except Exception as e:
    print(f"Error saving BERT model: {e}")

print("BERT fine-tuning script completed.")


#Loaded a pre-trained BERT model from Hugging Face.
#Loaded your preprocessed text dataset.
#Defined a classification head with softmax over emotion classes.
#Used Trainer API or manual training loop to train BERT on your dataset.
#Saved the fine-tuned model to your Models/ directory.