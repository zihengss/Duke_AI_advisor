import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Check for MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and tokenizer
model_path = "/Users/zihengs/Desktop/project/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load your dataset
dataset_path = "/Users/zihengs/Desktop/project/professors_QA_pairs.json"
dataset = load_dataset('json', data_files={'train': dataset_path, 'validation': dataset_path})

# Preprocess the dataset
def tokenize_function(examples):
    text = examples['output']  # Assuming 'text' is the key in your JSON file
    return tokenizer(text, padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(tokenized_datasets['train'])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory where the fine-tuned model will be saved
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    use_mps_device=True,
    # fp16=True, 
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
trainer.train()