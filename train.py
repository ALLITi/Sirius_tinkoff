import pandas as pd
from transformers import (GPT2Tokenizer, GPT2LMHeadModel, LineByLineTextDataset,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)

# 1. Load the data
data = pd.read_csv('data.csv')

# Combine the contexts for training
train_texts = data['context_3'].fillna('') + ' ' + data['context_2'].fillna('') + ' ' + data['context_1'].fillna('')
train_targets = data['response'].fillna('')

# Create a single text file for training data
with open('train_data.txt', 'w', encoding='utf-8') as f:
    for text, target in zip(train_texts, train_targets):
        f.write(text + ' ' + target + '\n')

# Initialize the GPT2 Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained("tinkoff-ai/ruDialoGPT-medium")
model = GPT2LMHeadModel.from_pretrained("tinkoff-ai/ruDialoGPT-medium")

# Split the data
data_len = len(train_texts)
train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='train_data.txt', block_size=128)[:int(0.8*data_len)]
test_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='train_data.txt', block_size=128)[int(0.8*data_len):]

# Prepare the datasets
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize the Trainer
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_steps=10,
    save_steps=10,
    warmup_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./custom_ruDialoGPT")