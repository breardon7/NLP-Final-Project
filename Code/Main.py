from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric

from Training import raw_dataset, tokenizer, tokenized_datasets, eval_dataloader, set_device, train_model_and_eval, metric, eval_batches, compute_metric

# Hyperparamters
model_name = AutoModelForSequenceClassification # model being used
optimizer_name = AdamW
LR = 5e-5
num_epochs = 3

# -----------------------Sequence Classification-----------------------
print('-----------------------Sequence Classification-----------------------')

# Task variables
dataset = "mrpc" # dataset being imported

# Preprocess dataset
tokenized_datasets = tokenized_datasets(dataset).remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Train batches
train_dataloader = DataLoader(tokenized_datasets["train"],
                                  shuffle=True, batch_size=8, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

# Model
model = model_name.from_pretrained(checkpoint, num_labels=2) # Identify model
outputs = model(batch) # Run batch through model to create outputs
print(outputs.loss, outputs.logits.shape)
optimizer = optimizer_name(model.parameters(), lr=LR) # Assign optimizer
num_training_steps = num_epochs * len(train_dataloader()) # Number of training steps
lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps) # Leanring rate schedular
print(num_training_steps)

set_device() # Set device - cuda if available, otherwise cpu
progress_bar = tqdm(range(num_training_steps)) # Model progress bar

train_model_and_eval(model) # Train and evaluate model
eval_dataloader = DataLoader(tokenized_datasets(dataset)["validation"],
                             batch_size=8, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))
eval_batches() # Eval Batches
compute_metric() # Compute Metric

