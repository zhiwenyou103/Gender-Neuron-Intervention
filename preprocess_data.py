"""Preprocess a gender-labeled CSV into token-id tensors for activation collection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from gender_neuron_utils import activation_file_tag, normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize gender-labeled text and save one tensor per gender/style label."
    )
    parser.add_argument("-m", "--model", default="meta-llama/Llama-2-7b-hf", help="Hugging Face model name or path.")
    parser.add_argument("-d", "--data_path", required=True, help="CSV file containing text and gender labels.")
    parser.add_argument("--text_column", default="Sentences", help="CSV column containing source text.")
    parser.add_argument("--gender_column", default="Gender", help="CSV column containing gender/style labels.")
    parser.add_argument("-o", "--output_dir", default="processed_data", help="Directory for token-id tensors.")
    parser.add_argument("--max_length", type=int, default=1024, help="Tokenizer sequence length.")
    parser.add_argument("--batch_size", type=int, default=512, help="Tokenizer batch size.")
    parser.add_argument(
        "--model_tag",
        default="",
        help="Optional override for file suffixes. Defaults to llama, bloom, or the sanitized model key for Qwen.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to the tokenizer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_tag = args.model_tag or activation_file_tag(args.model)

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Reading {args.data_path}...")
    frame = pd.read_csv(args.data_path)
    _validate_columns(frame, args.text_column, args.gender_column)

    grouped: Dict[str, List[str]] = defaultdict(list)
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Reading rows"):
        text = str(row[args.text_column]).strip() if pd.notna(row[args.text_column]) else ""
        label = normalize_label(row[args.gender_column]) if pd.notna(row[args.gender_column]) else ""
        if text and label:
            grouped[label].append(text)

    if not grouped:
        raise ValueError("No valid text/label pairs were found in the CSV.")

    metadata = {
        "model": args.model,
        "model_tag": file_tag,
        "source_csv": str(Path(args.data_path).resolve()),
        "text_column": args.text_column,
        "gender_column": args.gender_column,
        "max_length": args.max_length,
        "labels": {},
    }

    print(f"Found {len(grouped)} labels: {', '.join(sorted(grouped))}")
    for label in sorted(grouped):
        texts = grouped[label]
        tensor = tokenize_texts(tokenizer, texts, args.max_length, args.batch_size)
        output_path = output_dir / f"id.{label}.train.{file_tag}"
        torch.save(tensor, output_path)
        metadata["labels"][label] = {"num_examples": len(texts), "file": output_path.name}
        print(f"Saved {label}: {tuple(tensor.shape)} -> {output_path}")

    metadata_path = output_dir / f"preprocess_metadata.{file_tag}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {metadata_path}")


def tokenize_texts(tokenizer, texts: List[str], max_length: int, batch_size: int) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", leave=False):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        chunks.append(encoded["input_ids"].to(dtype=torch.long, device="cpu"))
    return torch.cat(chunks, dim=0)


def _validate_columns(frame: pd.DataFrame, text_column: str, gender_column: str) -> None:
    missing = [column for column in (text_column, gender_column) if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Available columns: {frame.columns.tolist()}"
        )


if __name__ == "__main__":
    main()
