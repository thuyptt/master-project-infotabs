#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INFOTABS Train & Eval Pipeline (DistilBERT • FLAN-T5-small • TAPAS-small)

- DistilBERT: paragraph premise + hypothesis → 3-class classification
- FLAN-T5-small: seq2seq (text-to-text) → "entailment"/"contradiction"/"neutral"
- TAPAS-small: table + question → 3-class classification

Requirements:
  pip install transformers datasets accelerate pandas scikit-learn beautifulsoup4 lxml
"""

import os, sys, argparse, json, re, torch
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    default_data_collator,
    TapasTokenizer,
    TapasForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ------------------ Constants ------------------

# for 3 class classification define the class labels and their IDs as follows
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# lists of possible column names to guess from the TSV files
COLUMN_GUESS = {
    "premise": ["premise", "prem", "paragraph", "premise_text"],
    "hypothesis": ["hypothesis", "hypo", "hypothesis_text", "hyp"],
    "label": ["label", "gold_label", "y"],
    "table_id": ["table_id", "tid", "table", "tableId"]
}

# ------------------ Helpers ------------------

# find the matching column in the TSV file (e.g. "premise") based on COLUMN_GUESS lists
# first: exact match, then: lower-case mathch
def guess_col(df: pd.DataFrame, kind: str):
    candidates = COLUMN_GUESS.get(kind, [])
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None

# read TSV file into DataFrame and fill missing values with empty strings so that tokenizers don't crash
def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")

# ensure label strings are standardized to {"entailment", "neutral", "contradiction"} and mapped to IDs
def ensure_label_ids(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df["_label_str"] = df[label_col].str.strip().str.lower()
    short_to_str = {"e": "entailment", "n": "neutral", "c": "contradiction"}
    df["_label_str"] = df["_label_str"].replace(short_to_str)
    repl = {"entails": "entailment", "entailed": "entailment", "contradict": "contradiction"}
    df["_label_str"] = df["_label_str"].replace(repl)
    if not set(df["_label_str"].unique()) <= set(LABEL2ID.keys()):
        bad = set(df["_label_str"].unique()) - set(LABEL2ID.keys())
        raise ValueError(f"Unknown labels: {bad}")
    df["_label_id"] = df["_label_str"].map(LABEL2ID)
    return df

# ------------------ Build Dataset ------------------

# for TAPAS-small we need to use the raw data from data/maindata
split_map = {
    'train': "infotabs_train.tsv",
    'dev': "infotabs_dev.tsv",
    'test_alpha1': "infotabs_test_alpha1.tsv",  
    'test_alpha2': "infotabs_test_alpha2.tsv",
    'test_alpha3': "infotabs_test_alpha3.tsv",
}


# load the TSV files for all desired splits (train/dev/test_alpha1/2/3) 
# find premis, hypothesis, label columns 
# create internal columns: - '_premise': text of linearized table, 
#                          - '_hypothesis': hypothesis text, 
#                          - '_label_str': '-label_id": normalized label strings and numeric IDs, 
#                          -  '_table_str': link to table 
# return dict of DataFrames
def build_dataset_from_tsvs(tsv_dir: Path, splits):
    data = {}
    for sp in splits:
        p = tsv_dir / split_map.get(sp, f"{sp}.tsv") # use the raw data from data/maindata
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
        df = read_tsv(p)
        prem_col = guess_col(df, "premise")
        hypo_col = guess_col(df, "hypothesis")
        label_col = guess_col(df, "label")
        df = ensure_label_ids(df, label_col)
        if prem_col is None:
            df["_premise"] = ""
        else:
            df["_premise"] = df[prem_col].astype(str)
        table_col = guess_col(df, "table_id")
        df["_table_id"] = df[table_col].astype(str) if table_col is not None else ""
        df["_hypothesis"] = df[hypo_col].astype(str)
        df = df[["_premise","_hypothesis","_label_str","_label_id","_table_id"]]
        data[sp] = df
    return data

# ------------------ TAPAS Table Helpers ------------------

# converts HTML table to key-value DataFrame with columns "key" and "value"9
def html_table_to_kv_dataframe(html_path: Path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th","td"])
        if len(cells) >= 2:
            key = cells[0].get_text(" ", strip=True)
            val = " ".join([c.get_text(' ', strip=True) for c in cells[1:]])
            rows.append((key, val))
    return pd.DataFrame(rows, columns=["key","value"])

# converts JSON object (dict) to key-value DataFrame with columns "key" and "value"
def json_table_to_kv_dataframe(json_path: Path):
    obj = json.loads(Path(json_path).read_text(encoding="utf-8", errors="ignore"))
    if isinstance(obj, dict):
        rows = [(str(k), str(v)) for k,v in obj.items()]
        return pd.DataFrame(rows, columns=["key","value"])
    raise ValueError(f"Unrecognized JSON layout: {json_path}")

# loads table by ID from tables_dir (either HTML or JSON) and converts to key-value DataFrame
def load_table_df(tables_dir: Path, table_id: str):
    tid = table_id if table_id.startswith("T") else f"T{table_id}"
    json_path = tables_dir / "json" / f"{tid}.json"
    html_path = tables_dir / "html" / f"{tid}.html"
    if json_path.exists():
        return json_table_to_kv_dataframe(json_path)
    if html_path.exists():
        return html_table_to_kv_dataframe(html_path)
    raise FileNotFoundError(f"Table not found for {tid}")

# ------------------ Encoders ------------------

# tokenizer receives a premise (paragraph or structured object) and hzpothesis (text), and generates input IDs and appends labels
def encode_distilbert(tokenizer, df, max_len):
    enc = tokenizer(list(df["_premise"]), list(df["_hypothesis"]),
                    truncation=True, max_length=max_len)
    enc["labels"] = df["_label_id"].tolist()
    return enc

# formulates classification as text-to-text task:
# Input: "Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
# Target: "entailment"/"contradiction"/"neutral"
def encode_flan_t5(tokenizer, df, max_len):
    inputs = [f"Premise: {p}\nHypothesis: {h}\nLabel:" for p,h in zip(df["_premise"], df["_hypothesis"])]
    targets = df["_label_str"].tolist()
    model_inputs = tokenizer(inputs, truncation=True, max_length=max_len)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, max_length=3)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# loads table by ID from tables_dir -> key-value DataFrame 
# tokenizer receives a table (DataFrame) and hypothesis (text), and generates input IDs and appends labels
def encode_tapas(tokenizer, df, tables_dir: Path, max_rows=64):
    feats = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "labels": []}
    for _, row in df.iterrows():
        tdf = load_table_df(tables_dir, row["_table_id"]).iloc[:max_rows]
        enc = tokenizer(table=tdf, queries=row["_hypothesis"], return_tensors="np", padding="max_length", max_length=512, truncation=True)
        feats["input_ids"].append(enc["input_ids"][0])
        feats["attention_mask"].append(enc["attention_mask"][0])
        feats["token_type_ids"].append(enc["token_type_ids"][0])
        feats["labels"].append(int(row["_label_id"]))
    return feats

# ------------------ Metrics ------------------

# compute accuracy and macro F1 score
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    y_pred = preds.argmax(-1)
    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

# ------------------ Training Loop ------------------

def train_and_eval(args):
    set_seed(args.seed) # set seed for reproducibility
    dfs = build_dataset_from_tsvs(Path(args.data_tsv_dir), args.splits) # load TSV files into DataFrames
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # initialize model, tokenizer, encode datasets, and data collator based on model type
    if args.model == "distilbert":
        model_name = "distilbert-base-uncased"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        feats = {sp: encode_distilbert(tok, df, args.max_len) for sp,df in dfs.items()}
        collate = DataCollatorWithPadding(tok)

    elif args.model == "flan-t5-small":
        model_name = "google/flan-t5-small"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        feats = {sp: encode_flan_t5(tok, df, args.max_len) for sp,df in dfs.items()}
        collate = DataCollatorForSeq2Seq(tok, model=model)

    elif args.model == "tapas-small":
        model_name = "google/tapas-small"
        tok = TapasTokenizer.from_pretrained(model_name)
        model = TapasForSequenceClassification.from_pretrained(model_name, num_labels=3)
        feats = {sp: encode_tapas(tok, df, Path(args.tables_dir)) for sp,df in dfs.items()}
        collate = None

    else:
        raise ValueError("Unknown model")

    ds_dict = {sp: Dataset.from_dict(feat) for sp,feat in feats.items()}
    dsd = DatasetDict(ds_dict)

    # define train and eval splits
    train_split = "train" if "train" in dsd else list(dsd.keys())[0]
    eval_split = "dev" if "dev" in dsd else list(dsd.keys())[0]

    # define training arguments
    ta = TrainingArguments(
        output_dir=str(out_dir/"ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=args.seed,
        report_to=[],
    )

    # initialize Trainer (no tokenizer for TAPAS as it is handled in the encode function)
    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=dsd[train_split] if args.mode=="train" else None,
        eval_dataset=dsd[eval_split],
        tokenizer=None if args.model=="tapas-small" else tok,
        data_collator=default_data_collator, #collate,
        compute_metrics=compute_metrics,
    )

    # Train and save model
    if args.mode == "train":
        trainer.train()
        trainer.save_model(str(out_dir/"ckpt"/"best"))
        torch.cuda.empty_cache() # free up GPU memory after evaluation + saving metric results

    # Evaluate: evaluate on all specified splits and save metrics to JSON files
    metrics_summary = {}
    for sp in args.splits:
        metrics = trainer.evaluate(eval_dataset=dsd[sp])
        metrics_summary[sp] = metrics
        with open(out_dir/f"metrics_{sp}.json","w") as f:
            json.dump(metrics, f, indent=2)
            torch.cuda.empty_cache() # free up GPU memory after evaluation + saving metric results
    with open(out_dir/"metrics_summary.json","w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(json.dumps(metrics_summary, indent=2))

# ------------------ CLI (Command Line Inference/Kommandizeilen-Schnittstelle) ------------------

# define command line arguments and call train_and_eval
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    choices=["distilbert","flan-t5-small","tapas-small"])
    ap.add_argument("--mode", type=str, default="train", choices=["train","test"])
    ap.add_argument("--splits", nargs="+", default=["train","dev","test_alpha1","test_alpha2","test_alpha3"])
    ap.add_argument("--data_tsv_dir", type=str, required=True)
    ap.add_argument("--tables_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=16) # or what your GPU allows
    ap.add_argument("--epochs", type=int, default=4) # optimal: 3-5
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    train_and_eval(args)

if __name__ == "__main__":
    main()