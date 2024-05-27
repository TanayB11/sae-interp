import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

def get_tokenizer_and_loaders(cfg):
    # data preprocessing from https://github.com/rajpurkar/cs197-lec4/blob/master/demo.ipynb
    device = cfg.device

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    tokenizer.pad_token = tokenizer.eos_token

    def add_eos(example):
        example['text'] = example['text'] + tokenizer.eos_token
        return example

    def tokenize_datasets(examples):
        return tokenizer(examples['text'], truncation=True)

    def group_texts(examples):
        # group texts into blocks of block_size
        block_size = cfg.batch_size

        # repeat concatenation for input_ids and other keys
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size

        # populate each of input_ids and other keys 
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # add labels because we'll need it as the output
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset = load_dataset('roneneldan/TinyStories', split='train').shuffle(seed=cfg.seed).select(range(cfg.trainset_size))
    val_dataset = load_dataset('roneneldan/TinyStories', split='validation')

    train_dataset = train_dataset.map(add_eos, num_proc=4)
    val_dataset = val_dataset.map(add_eos, num_proc=4)

    train_dataset = train_dataset.map(tokenize_datasets, num_proc=4, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_datasets, num_proc=4, remove_columns=['text'])

    train_dataset = train_dataset.map(
        group_texts,
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=4,
    ).with_format('torch', device=device)

    val_dataset = val_dataset.map(
        group_texts,
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=4,
    ).with_format('torch', device=device)

    # already shuffled earlier
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    return tokenizer, train_loader, val_loader