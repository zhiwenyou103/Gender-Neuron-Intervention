"""Shared utilities for gender-neuron identification experiments."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


TEXT_COLUMN_CANDIDATES = (
    "text",
    "texts",
    "sentence",
    "sentences",
    "prompt",
    "prompts",
    "content",
    "contents",
    "input",
    "inputs",
    "utterance",
    "utterances",
)

GENDER_COLUMN_CANDIDATES = ("gender", "sex", "label", "style")


def model_key(model_name: str) -> str:
    """Return a filesystem-safe key for model-specific outputs."""
    return model_name.split("/")[-1].lower().replace("-", "_")


def model_family(model_name: str) -> str:
    """Return the supported model family used by the vLLM patching code."""
    lower = model_name.lower()
    if "llama" in lower:
        return "llama"
    if "qwen" in lower:
        return "qwen"
    if "bloom" in lower:
        return "bloom"
    raise ValueError(
        f"Unsupported model family for '{model_name}'. "
        "This repo currently supports Llama-like, Qwen-like, and BLOOM models."
    )


def is_llama_like(model_name: str) -> bool:
    return model_family(model_name) in {"llama", "qwen"}


def activation_file_tag(model_name: str) -> str:
    """Return the suffix used for token-id and activation files."""
    family = model_family(model_name)
    if family == "llama":
        return "llama"
    if family == "bloom":
        return "bloom"
    return model_key(model_name)


def normalize_label(value: object) -> str:
    return str(value).strip().lower()


def case_insensitive_lookup(fieldnames: List[str], name: Optional[str]) -> Optional[str]:
    if not fieldnames or not name:
        return None
    mapping = {str(field).strip().lower(): field for field in fieldnames}
    return mapping.get(str(name).strip().lower())


def read_labeled_csv(
    csv_path: str | os.PathLike[str],
    text_column: Optional[str] = None,
    gender_column: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Read text and optional gender labels from a CSV file.

    Headered CSVs are preferred. If no header is detected, the first column is
    treated as text and the second column, when present, as the label.
    """
    path = Path(csv_path)
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = True

        if not has_header:
            try:
                first_row = next(row for row in csv.reader(handle) if any(cell.strip() for cell in row))
                header_tokens = set(TEXT_COLUMN_CANDIDATES) | set(GENDER_COLUMN_CANDIDATES)
                has_header = any(cell.strip().lower() in header_tokens for cell in first_row)
            except StopIteration:
                return rows
            finally:
                handle.seek(0)

        if has_header:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            resolved_text_column = _resolve_column(fieldnames, text_column, TEXT_COLUMN_CANDIDATES)
            resolved_gender_column = _resolve_column(fieldnames, gender_column, GENDER_COLUMN_CANDIDATES)

            if resolved_text_column is None:
                raise ValueError(
                    f"Could not find a text column in {path}. "
                    f"Available columns: {fieldnames}"
                )

            for row in reader:
                text = str(row.get(resolved_text_column, "") or "").strip()
                if not text:
                    continue
                gender = ""
                if resolved_gender_column is not None:
                    gender = normalize_label(row.get(resolved_gender_column, ""))
                rows.append({"text": text, "gender": gender})
        else:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                text = str(row[0] or "").strip()
                if not text:
                    continue
                gender = normalize_label(row[1]) if len(row) > 1 else ""
                rows.append({"text": text, "gender": gender})

    return rows


def preferred_gender_order(labels: List[str]) -> List[str]:
    preferred = ["male", "female", "neutral", "masculine", "feminine", "gender-neutral"]
    return [label for label in preferred if label in labels] + sorted(
        label for label in labels if label not in preferred
    )


def infer_label_from_activation_path(path: str | os.PathLike[str], file_tag: str) -> Optional[str]:
    escaped_tag = re.escape(file_tag)
    match = re.match(rf"activation\.(.+)\.train\.{escaped_tag}$", Path(path).name)
    return match.group(1) if match else None


def _resolve_column(
    fieldnames: List[str],
    requested: Optional[str],
    candidates: tuple[str, ...],
) -> Optional[str]:
    if requested:
        resolved = case_insensitive_lookup(fieldnames, requested)
        if resolved is None:
            raise ValueError(
                f"Column '{requested}' was not found. Available columns: {fieldnames}"
            )
        return resolved

    for candidate in candidates:
        resolved = case_insensitive_lookup(fieldnames, candidate)
        if resolved is not None:
            return resolved
    return fieldnames[0] if fieldnames and candidates == TEXT_COLUMN_CANDIDATES else None
