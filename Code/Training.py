from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric

# num_epochs = 3
# dataset = "mrpc"
checkpoint = "bert-base-uncased"

def tokenizer():
    return AutoTokenizer.from_pretrained(checkpoint)

def raw_dataset(dataset):
    raw_dataset = load_dataset("glue", dataset)
    return raw_dataset

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def tokenized_datasets(dataset):
    tokenized_datasets = raw_dataset(dataset).map(tokenize_function, batched=True)
    return tokenized_datasets

# tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# tokenized_datasets.set_format("torch")
# tokenized_datasets["train"].column_names

'''def train_dataloader():
    train_dataloader = DataLoader(tokenized_datasets["train"],
                                  shuffle=True, batch_size=8, collate_fn=data_collator)
    return train_dataloader'''

def eval_dataloader():
    eval_dataloader = DataLoader(tokenized_datasets["validation"],
                             batch_size=8, collate_fn=data_collator)
    return eval_dataloader

# Train batches
'''for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

'''
# model_name = AutoModelForSequenceClassification
# model = model_name.from_pretrained(checkpoint, num_labels=2)


# outputs = model(batch)
# print(outputs.loss, outputs.logits.shape)
# optimizer = AdamW(model.parameters(), lr=5e-5)

# num_training_steps = num_epochs * len(train_dataloader())
# lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                              #num_warmup_steps=0, num_training_steps=num_training_steps)
# print(num_training_steps)

def set_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    return device

#progress_bar = tqdm(range(num_training_steps))

#model.train()

def train_model_and_eval(model):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader():
            batch = {k: v.to(set_device()) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    model.eval()

def metric(dataset):
        metric = load_metric("glue", dataset)
        return metric

def eval_batches():
    for batch in eval_dataloader:
        batch = {k: v.to(set_device()) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric().add_batch(predictions=predictions, references=batch["labels"])

def compute_metric():
    metric().compute()