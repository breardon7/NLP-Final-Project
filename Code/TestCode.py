from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import DebertaForSequenceClassification
from transformers import RobertaForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric



glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
num_epochs = 3
batch_size = 10
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = RobertaForSequenceClassification.from_pretrained(checkpoint)

for dset in glue_datasets:
    if dset == 'mrpc':
        raw_datasets = load_dataset("glue", "mrpc")

        def tokenize_function(example):
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        print(tokenized_datasets)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        tokenized_datasets["train"].column_names

        train_dataloader = DataLoader(tokenized_datasets["train"],
                                      shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        eval_dataloader = DataLoader(tokenized_datasets["validation"],
                                     batch_size=batch_size, collate_fn=data_collator)

        for batch in train_dataloader:
            break
        {k: v.shape for k, v in batch.items()}

        model = model
        outputs = model(**batch)
        print(outputs.loss, outputs.logits.shape)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                                      num_warmup_steps=0, num_training_steps=num_training_steps)
        print(num_training_steps)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        device

        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        metric = load_metric("glue", "mrpc")
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        score = metric.compute()
        print(score, 'mrpc')

    elif dset == 'skip_rte':
        raw_datasets = load_dataset("glue", "rte")
        print(raw_datasets)
        checkpoint = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = DebertaForSequenceClassification.from_pretrained(checkpoint)


        def tokenize_function(example):
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        print(tokenized_datasets)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        tokenized_datasets["train"].column_names

        train_dataloader = DataLoader(tokenized_datasets["train"],
                                      shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        eval_dataloader = DataLoader(tokenized_datasets["validation"],
                                     batch_size=batch_size, collate_fn=data_collator)

        for batch in train_dataloader:
            break
        {k: v.shape for k, v in batch.items()}

        model = model
        outputs = model(**batch)
        print(outputs.loss, outputs.logits.shape)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                     num_warmup_steps=0, num_training_steps=num_training_steps)
        print(num_training_steps)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        device

        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        metric = load_metric("glue", "rte")
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        score = metric.compute()
        print(score)

    elif dset == 'skip_wnli':
        raw_datasets = load_dataset("glue", "wnli")
        print(raw_datasets)
        checkpoint = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = DebertaForSequenceClassification.from_pretrained(checkpoint)


        def tokenize_function(example):
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        print(tokenized_datasets)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        tokenized_datasets["train"].column_names

        train_dataloader = DataLoader(tokenized_datasets["train"],
                                      shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        eval_dataloader = DataLoader(tokenized_datasets["validation"],
                                     batch_size=batch_size, collate_fn=data_collator)

        for batch in train_dataloader:
            break
        {k: v.shape for k, v in batch.items()}

        model = model
        outputs = model(**batch)
        print(outputs.loss, outputs.logits.shape)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                     num_warmup_steps=0, num_training_steps=num_training_steps)
        print(num_training_steps)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        device

        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        metric = load_metric("glue", "wnli")
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        score = metric.compute()
        print(score)

    elif dset == 'skip_sst2':
        raw_datasets = load_dataset("glue", "sst2")
        print(raw_datasets)
        checkpoint = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = DebertaForSequenceClassification.from_pretrained(checkpoint)


        def tokenize_function(example):
            return tokenizer(example["sentence"], truncation=True)


        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        print(tokenized_datasets)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        tokenized_datasets["train"].column_names

        train_dataloader = DataLoader(tokenized_datasets["train"],
                                      shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        eval_dataloader = DataLoader(tokenized_datasets["validation"],
                                     batch_size=batch_size, collate_fn=data_collator)

        for batch in train_dataloader:
            break
        {k: v.shape for k, v in batch.items()}

        model = model
        outputs = model(**batch)
        print(outputs.loss, outputs.logits.shape)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                     num_warmup_steps=0, num_training_steps=num_training_steps)
        print(num_training_steps)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        device

        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        metric = load_metric("glue", "sst2")
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        score = metric.compute()
        print(score)

    '''
    elif dset == 'cola': Rohan
    elif dset == 'qqp': Rohan
    elif dset == 'stsb': JC
    elif dset == 'mnli': Nigel
    elif dset == 'qnli': JC
    elif: # ax