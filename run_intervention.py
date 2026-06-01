"""Generate baseline and neuron-intervention outputs for a gender-labeled CSV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("VLLM_USE_V1", "0")


def _set_visible_gpu_from_argv() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=None)
    args, _ = parser.parse_known_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


_set_visible_gpu_from_argv()

import torch
from vllm import LLM, SamplingParams

from gender_neuron_utils import model_key, read_labeled_csv
from vllm_mlp_utils import attach_mlp_masks, restore_mlp_forwards


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{text}\n\n"
    "### Response:"
)

STRICT_INSTRUCTIONS = {
    "male": (
        "Please transfer the following sentence into a tone stereotypically associated with males while "
        "maintaining the original meaning. Avoid adding explicit gender identifiers; reflect the style via tone, "
        "word choice, and perspective. Return ONLY the rewritten sentence."
    ),
    "female": (
        "Please transfer the following sentence into a tone stereotypically associated with females while "
        "maintaining the original meaning. Avoid adding explicit gender identifiers; reflect the style via tone, "
        "word choice, and perspective. Return ONLY the rewritten sentence."
    ),
    "neutral": (
        "Please transfer the following sentence into a gender-neutral tone while maintaining the original meaning. "
        "Avoid gendered terms unless necessary; prefer neutral occupational/role nouns and pronouns. "
        "Return ONLY the rewritten sentence."
    ),
}

LEXICON_INSTRUCTIONS = {
    "male": (
        "Rewrite the sentence with wording stereotypically associated with male tone. It is acceptable to use "
        "subtle gender-coded lexical choices and role nouns that suggest male perspective when natural. "
        "Return ONLY the rewritten sentence."
    ),
    "female": (
        "Rewrite the sentence with wording stereotypically associated with female tone. It is acceptable to use "
        "subtle gender-coded lexical choices and role nouns that suggest female perspective when natural. "
        "Return ONLY the rewritten sentence."
    ),
    "neutral": (
        "Rewrite the sentence in a gender-neutral tone, using neutral role nouns and pronouns where possible. "
        "Return ONLY the rewritten sentence."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline generation and masked MLP-neuron intervention generation."
    )
    parser.add_argument("--input_csv", required=True, help="CSV containing source text and optional gender labels.")
    parser.add_argument("--csv_text_column", default=None, help="Text column. Auto-detected when omitted.")
    parser.add_argument("--csv_gender_column", default=None, help="Gender column. Auto-detected when omitted.")
    parser.add_argument("-m", "--model", default="meta-llama/Llama-2-7b-hf", help="Hugging Face model name or path.")
    parser.add_argument("--activation_mask", default="", help="Explicit path to a saved activation mask.")
    parser.add_argument("--mask_dir", default="activation_mask", help="Directory used for auto-loading exclusive masks.")
    parser.add_argument("--mask_suffix", default="", help="Optional suffix for mask and gender-map files.")
    parser.add_argument("--mask_factor", type=float, default=1.0, help="Suppression strength. 1.0 zeros masked neurons.")
    parser.add_argument("--boost_factor", type=float, default=1.0, help="Optional amplification for kept-gender neurons.")
    parser.add_argument("--mask_at_product", action="store_true", help="Apply Llama/Qwen masks after gate*up instead of on gate activations.")
    parser.add_argument("--output_root", default="results", help="Root directory for JSONL outputs.")
    parser.add_argument("--output_suffix", default="", help="Optional suffix for the output directory.")
    parser.add_argument("--prompt_variant", choices=["strict", "lexicon", "none"], default="strict", help="Prompt style.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_rows", type=int, default=0, help="Optional row cap for debugging.")
    parser.add_argument("--tensor_parallel_size", type=int, default=0, help="vLLM tensor parallel size. Defaults to vLLM's default.")
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU index to expose before vLLM imports.")
    parser.add_argument("--disable_custom_all_reduce", action="store_true", help="Forwarded to vLLM for GPU P2P compatibility.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Forwarded to vLLM model loading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    mask_path = resolve_mask_path(args)
    activation_masks, genders = load_masks_and_genders(mask_path, args.mask_suffix)
    print(f"Loaded mask: {mask_path}")
    for label, layer_masks in zip(genders, activation_masks):
        print(f"  {label}: {sum(len(layer) for layer in layer_masks):,} neurons")

    rows = read_labeled_csv(args.input_csv, args.csv_text_column, args.csv_gender_column)
    if args.num_rows > 0:
        rows = rows[: args.num_rows]
    if not rows:
        raise ValueError(f"No usable rows were loaded from {args.input_csv}.")
    print(f"Loaded {len(rows)} rows from {args.input_csv}")

    model_kwargs = {
        "model": args.model,
        "enforce_eager": True,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.tensor_parallel_size > 0:
        model_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    model = LLM(**model_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=1.1,
        stop=stop_sequences(),
    )

    prompts, prompt_map = build_prompts(rows, genders, args.prompt_variant)
    print("Generating baseline outputs...")
    baseline_outputs = collect_outputs(model.generate(prompts, sampling_params), prompt_map, len(rows))

    masked_outputs: Dict[str, Dict[int, Dict[str, str]]] = {
        keep_gender: {row_idx: {} for row_idx in range(len(rows))} for keep_gender in genders
    }
    for keep_gender in genders:
        print(f"Generating masked outputs while keeping '{keep_gender}' neurons...")
        zero_masks, boost_masks = build_keep_and_zero_masks(activation_masks, genders, keep_gender)
        originals = attach_mlp_masks(
            model,
            args.model,
            zero_masks,
            boost_masks,
            args.mask_factor,
            args.boost_factor,
            args.mask_at_product,
        )
        try:
            outputs = model.generate(prompts, sampling_params)
            masked_outputs[keep_gender] = collect_outputs(outputs, prompt_map, len(rows))
        finally:
            restore_mlp_forwards(model, args.model, originals)

    output_path = write_results(args, rows, genders, baseline_outputs, masked_outputs, mask_path)
    print(f"Saved per-row outputs to {output_path}")


def resolve_mask_path(args: argparse.Namespace) -> Path:
    if args.activation_mask:
        path = Path(args.activation_mask)
    else:
        path = Path(args.mask_dir) / f"{model_key(args.model)}_exclusive{args.mask_suffix}"
    if not path.is_file():
        raise FileNotFoundError(
            f"Mask file not found: {path}. Run identify_gender_neurons.py first or pass --activation_mask."
        )
    return path


def load_masks_and_genders(mask_path: Path, suffix: str) -> tuple[List[List[torch.Tensor]], List[str]]:
    masks = torch.load(mask_path, map_location="cpu")
    gender_map_path = mask_path.parent / f"gender_map{suffix}.txt"
    if not gender_map_path.is_file():
        gender_map_path = mask_path.parent / "gender_map.txt"

    genders: List[str] = []
    if gender_map_path.is_file():
        mapping = {}
        for line in gender_map_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            idx, label = line.split(":", 1)
            mapping[int(idx.strip())] = label.strip()
        genders = [label for _, label in sorted(mapping.items())]

    if not genders:
        fallback = ["male", "female", "neutral"]
        genders = fallback[: len(masks)]
    if len(genders) != len(masks):
        raise ValueError(f"Gender map has {len(genders)} labels but mask has {len(masks)} entries.")
    return masks, genders


def build_prompts(
    rows: List[Dict[str, str]],
    target_genders: List[str],
    prompt_variant: str,
) -> tuple[List[str], List[Tuple[int, str]]]:
    prompts: List[str] = []
    mapping: List[Tuple[int, str]] = []
    for row_idx, row in enumerate(rows):
        for target_gender in target_genders:
            prompts.append(render_prompt(row["text"], target_gender, prompt_variant))
            mapping.append((row_idx, target_gender))
    return prompts, mapping


def render_prompt(text: str, target_gender: str, prompt_variant: str) -> str:
    if prompt_variant == "none":
        return text
    family = canonical_gender_family(target_gender)
    instructions = LEXICON_INSTRUCTIONS if prompt_variant == "lexicon" else STRICT_INSTRUCTIONS
    instruction = instructions.get(
        family,
        f"Rewrite the sentence in a {target_gender} style while preserving its meaning. Return ONLY the rewritten sentence.",
    )
    return PROMPT_TEMPLATE.format(instruction=instruction, text=text)


def canonical_gender_family(label: str) -> str:
    normalized = label.lower().strip()
    if normalized in {"male", "masculine", "man"}:
        return "male"
    if normalized in {"female", "feminine", "woman"}:
        return "female"
    if normalized in {"neutral", "gender-neutral", "nonbinary", "non-binary"}:
        return "neutral"
    return normalized


def collect_outputs(outputs, prompt_map: List[Tuple[int, str]], row_count: int) -> Dict[int, Dict[str, str]]:
    collected: Dict[int, Dict[str, str]] = {row_idx: {} for row_idx in range(row_count)}
    for (row_idx, target_gender), output in zip(prompt_map, outputs):
        collected[row_idx][target_gender] = clean_generation(output.outputs[0].text)
    return collected


def clean_generation(text: str) -> str:
    markers = (
        "\n\nNote:",
        "\nNote:",
        "\n\nExplanation:",
        "\nExplanation:",
        "\n\nThe rewritten",
        "\nThe rewritten",
        "\n\n###",
        "\n###",
        " (Note:",
    )
    cleaned = text.strip()
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned.removesuffix("###").strip()


def build_keep_and_zero_masks(
    activation_masks: List[List[torch.Tensor]],
    genders: List[str],
    keep_gender: str,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    gender_to_index = {label: idx for idx, label in enumerate(genders)}
    if keep_gender not in gender_to_index:
        raise ValueError(f"Unknown keep gender '{keep_gender}'. Available labels: {genders}")

    keep_idx = gender_to_index[keep_gender]
    num_layers = len(activation_masks[keep_idx])
    zero_masks: List[torch.Tensor] = []
    boost_masks: List[torch.Tensor] = []

    for layer_idx in range(num_layers):
        suppress = [
            activation_masks[gender_idx][layer_idx]
            for gender_idx in range(len(genders))
            if gender_idx != keep_idx and len(activation_masks[gender_idx][layer_idx]) > 0
        ]
        zero_masks.append(torch.unique(torch.cat(suppress)) if suppress else torch.tensor([], dtype=torch.long))
        keep_layer = activation_masks[keep_idx][layer_idx]
        boost_masks.append(keep_layer if len(keep_layer) > 0 else torch.tensor([], dtype=torch.long))

    return zero_masks, boost_masks


def write_results(
    args: argparse.Namespace,
    rows: List[Dict[str, str]],
    genders: List[str],
    baseline_outputs: Dict[int, Dict[str, str]],
    masked_outputs: Dict[str, Dict[int, Dict[str, str]]],
    mask_path: Path,
) -> Path:
    output_dir = Path(args.output_root) / model_key(args.model) / f"per_row_gender{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_suffix = "" if args.prompt_variant == "strict" else f"_{args.prompt_variant}"
    strength_suffix = "" if args.mask_factor == 1.0 else f"_strength{args.mask_factor:g}"
    boost_suffix = "" if args.boost_factor == 0.0 else f"_boost{args.boost_factor:g}"
    output_path = output_dir / f"per_row.contrast_exclusive{strength_suffix}{boost_suffix}{prompt_suffix}.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for row_idx, row in enumerate(rows):
            record = {
                "row_index": row_idx,
                "input_text": row.get("text", ""),
                "input_gender": row.get("gender", ""),
                "genders": genders,
                "baseline": {label: baseline_outputs[row_idx].get(label, "") for label in genders},
                "masked_keep": {
                    keep_gender: {
                        target_gender: masked_outputs[keep_gender][row_idx].get(target_gender, "")
                        for target_gender in genders
                    }
                    for keep_gender in genders
                },
                "mask_kind": "exclusive",
                "mask_path": str(mask_path),
                "mask_factor": args.mask_factor,
                "boost_factor": args.boost_factor,
                "prompt_variant": args.prompt_variant,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def stop_sequences() -> List[str]:
    return [
        "\n\nNote:",
        "\nNote:",
        "\n\nExplanation:",
        "\nExplanation:",
        "\n\nThe rewritten",
        "\nThe rewritten",
        "\n\n###",
        "\n###",
    ]


if __name__ == "__main__":
    main()
