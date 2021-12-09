from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric

from transformers import RobertaTokenizer,DataCollatorWithPadding # Tokenizer
from transformers import RobertaForSequenceClassification # Task

import os
import torch

# parameters
dataset = "qnli"
MODEL_TYPE = 'roberta-base'# roberta-base # microsoft/deberta-base
L_RATE = 5e-5
MAX_LEN = 256
NUM_EPOCHS = 2
BATCH_SIZE = 4
NUM_CORES = 4*torch.cuda.device_count() # num of gpu

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def tokenize_function(example):
    return tokenizer(example["question"], example["sentence"], truncation=True)

# Load the data
raw_datasets = load_dataset("glue", dataset)


# tokenize the data
tokenizer = RobertaTokenizer.from_pretrained(MODEL_TYPE)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# cleaning
tokenized_datasets = tokenized_datasets.remove_columns(["question", "sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names # ['attention_mask', 'input_ids', 'labels']


# data loading
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(tokenized_datasets["train"],
                              shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator) # ,  num_workers=NUM_CORES
eval_dataloader = DataLoader(tokenized_datasets["validation"],
                             batch_size=BATCH_SIZE, collate_fn=data_collator) # , num_workers=NUM_CORES

# check shapes of input
for batch in eval_dataloader:
    break
{k: v.shape for k, v in batch.items()}


# Model
model = RobertaForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels=2)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
optimizer = AdamW(model.parameters(), lr=L_RATE) # optimizer
num_training_steps = NUM_EPOCHS * len(train_dataloader) # Number of training steps
lr_scheduler = get_scheduler("linear", optimizer=optimizer,num_warmup_steps=0, num_training_steps=num_training_steps) # Leanring rate schedular
model.to(device) # model to gpu

# train
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(NUM_EPOCHS):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)

# model save
torch.save(model.state_dict(), "model_roberta_qnli.pt") # change the model name

# evaluation
metric = load_metric("glue", "qnli")

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
