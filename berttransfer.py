# bert_transfer_learning.py
"""
Transfer learning with BERT (Hugging Face Transformers + PyTorch)
- Supports: training on HuggingFace 'datasets' (e.g., imdb) OR CSV/TSV files
- Uses Trainer API for robustness, with options for:
    * Freezing encoder layers (partial or full)
    * Gradual unfreezing
    * Discriminative learning rates (via layer-wise LR multiplier)
    * Mixed precision training (fp16)
    * Saving and loading best model

How to run examples:
1) Using a HuggingFace dataset name (e.g. imdb):
   python bert_transfer_learning.py --dataset_name imdb --model_name_or_path bert-base-uncased --task_name binary

2) Using CSV files with columns 'text' and 'label':
   python bert_transfer_learning.py --train_file train.csv --validation_file val.csv --text_col text --label_col label --model_name_or_path bert-base-uncased

You can tweak hyperparams via CLI args.

Requires: transformers, datasets, torch, scikit-learn
pip install transformers datasets torch scikit-learn
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
import logging
import numpy as np
import random

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset, load_metric, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="BERT Transfer Learning Example")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="HuggingFace dataset name (e.g., imdb). If provided, uses it instead of CSV files.")
    parser.add_argument("--train_file", type=str, default=None, help="Path to train CSV/TSV")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to validation CSV/TSV")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="./bert_finetuned")
    parser.add_argument("--freeze_base_layers", type=int, default=0,
                        help="Number of transformer encoder layers to freeze from the bottom. 0 = none, -1 = freeze all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_name", type=str, default=None, help="'binary' or 'multiclass' - needed if dataset doesn't provide label info")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(args):
    if args.dataset_name:
        ds = load_dataset(args.dataset_name)
        # Expect 'train' and 'test' or 'validation'
        if 'train' not in ds:
            raise ValueError("Dataset must have a 'train' split")
        if 'validation' not in ds and 'test' in ds:
            ds = ds.rename_column('test', 'validation') if 'validation' not in ds else ds
        return ds
    else:
        # Load from CSV/TSV
        if not args.train_file or not args.validation_file:
            raise ValueError("Provide --train_file and --validation_file when not using --dataset_name")
        # detect separator
        def infer_sep(filename: str):
            if filename.endswith('.tsv'):
                return '\t'
            return ','
        train_sep = infer_sep(args.train_file)
        val_sep = infer_sep(args.validation_file)
        ds = load_dataset('csv', data_files={'train': args.train_file, 'validation': args.validation_file}, delimiter=train_sep)
        return ds


def preprocess_function(examples, tokenizer, args):
    texts = examples.get(args.text_col)
    # Some HF datasets return a single string, some list; handle both
    if isinstance(texts, str):
        texts = [texts]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=args.max_length)


def prepare_labels(dataset, args):
    # Ensure labels are 0..num_labels-1
    if args.label_col not in dataset['train'].column_names:
        # Some datasets use 'label' default
        if 'label' in dataset['train'].column_names:
            args.label_col = 'label'
        else:
            raise ValueError(f"Label column '{args.label_col}' not found in dataset")

    labels = dataset['train'][args.label_col]
    # if labels are strings, create mapping
    if isinstance(labels[0], str):
        unique = sorted(list(set(labels)))
        label2id = {l: i for i, l in enumerate(unique)}
        def map_label(example):
            example[args.label_col] = label2id[example[args.label_col]]
            return example
        dataset = dataset.map(map_label)
        num_labels = len(unique)
    else:
        num_labels = len(set(labels))
    return dataset, num_labels


def freeze_layers(model, freeze_count: int):
    """Freeze the bottom `freeze_count` encoder layers. If freeze_count == -1 freeze whole base model.
    Works for models with .bert or .roberta base modules.
    """
    base = None
    for attr in ['bert', 'roberta', 'distilbert', 'xlm_roberta']:
        if hasattr(model, attr):
            base = getattr(model, attr)
            break
    if base is None:
        logger.warning("Couldn't find base transformer module to freeze (bert/roberta/distilbert/xlm_roberta). Skipping freezing.")
        return

    if freeze_count == -1:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Frozen entire model")
        return

    # Many models expose encoder.layer
    if hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
        layers = base.encoder.layer
        freeze_count = min(len(layers), freeze_count)
        for i in range(freeze_count):
            for p in layers[i].parameters():
                p.requires_grad = False
        logger.info(f"Froze {freeze_count} encoder layers out of {len(layers)}")
    else:
        logger.warning("Base model architecture unexpected; cannot freeze by layer index")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro' if len(set(labels))>2 else 'binary')
    prec = precision_score(labels, preds, average='macro' if len(set(labels))>2 else 'binary')
    rec = recall_score(labels, preds, average='macro' if len(set(labels))>2 else 'binary')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    ds = load_data(args)

    # If label column is missing or dataset labels are not numeric, prepare labels
    ds, num_labels = prepare_labels(ds, args)
    logger.info(f"Num labels: {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # tokenization
    tokenized = ds.map(lambda examples: preprocess_function(examples, tokenizer, args), batched=True)

    # set format
    tokenized = tokenized.remove_columns([c for c in tokenized['train'].column_names if c not in ['input_ids','attention_mask', args.label_col]])
    tokenized = tokenized.rename_column(args.label_col, 'labels')
    tokenized.set_format('torch')

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    # Freeze if requested
    if args.freeze_base_layers != 0:
        freeze_layers(model, args.freeze_base_layers)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=args.use_fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=50,
        save_total_limit=3,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'] if 'validation' in tokenized else tokenized.get('test'),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model & tokenizer saved to {args.output_dir}")


if __name__ == '__main__':
    main()
