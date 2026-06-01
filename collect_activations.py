"""Collect MLP activation statistics for one gender/style label with vLLM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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

from gender_neuron_utils import activation_file_tag
from vllm_mlp_utils import attach_activation_collectors, get_architecture_dimensions, restore_mlp_forwards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run token-id tensors through a model and collect MLP activation moments."
    )
    parser.add_argument("-m", "--model", default="meta-llama/Llama-2-7b-hf", help="Hugging Face model name or path.")
    parser.add_argument(
        "-g",
        "--gender",
        "--style",
        dest="gender",
        required=True,
        help="Gender/style label to process, matching id.<label>.train.<tag>.",
    )
    parser.add_argument("-d", "--data_dir", default="processed_data", help="Directory containing token-id tensors.")
    parser.add_argument(
        "--model_tag",
        default="",
        help="Optional file suffix override. Defaults to llama, bloom, or sanitized model key for Qwen.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=0,
        help="Optional sequence length for one-dimensional legacy tensors. Two-dimensional tensors are used as saved.",
    )
    parser.add_argument("--max_sequences", type=int, default=0, help="Optional cap for quick debugging runs.")
    parser.add_argument("--tensor_parallel_size", type=int, default=0, help="vLLM tensor parallel size. Defaults to all visible GPUs.")
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU index to expose before vLLM imports.")
    parser.add_argument("--disable_custom_all_reduce", action="store_true", help="Forwarded to vLLM for GPU P2P compatibility.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Forwarded to vLLM model loading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for vLLM activation collection.")

    data_dir = Path(args.data_dir)
    file_tag = args.model_tag or activation_file_tag(args.model)
    input_path = data_dir / f"id.{args.gender}.train.{file_tag}"
    if not input_path.is_file():
        raise FileNotFoundError(f"Token-id tensor not found: {input_path}")

    input_ids, token_count = load_input_ids(input_path, args.sequence_length, args.max_sequences)
    print(f"Loaded {tuple(input_ids.shape)} token ids from {input_path}")

    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()
    model = LLM(
        model=args.model,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        trust_remote_code=args.trust_remote_code,
    )
    num_layers, intermediate_size = get_architecture_dimensions(model, args.model)
    stats = {
        "sum1": torch.zeros(num_layers, intermediate_size, dtype=torch.float32, device="cuda"),
        "sum2": torch.zeros(num_layers, intermediate_size, dtype=torch.float32, device="cuda"),
        "over_zero": torch.zeros(num_layers, intermediate_size, dtype=torch.int64, device="cuda"),
    }

    originals = attach_activation_collectors(model, args.model, stats)
    try:
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=sampling_params)
    finally:
        restore_mlp_forwards(model, args.model, originals)

    output = {
        "n": int(token_count),
        "sum1": stats["sum1"].cpu(),
        "sum2": stats["sum2"].cpu(),
        "over_zero": stats["over_zero"].cpu(),
        "model": args.model,
        "gender": args.gender,
        "activation_site": "mlp_product_pre_down_projection",
    }
    output_path = data_dir / f"activation.{args.gender}.train.{file_tag}"
    torch.save(output, output_path)
    print(f"Saved activation statistics to {output_path}")


def load_input_ids(
    path: Path,
    sequence_length: int,
    max_sequences: int,
) -> tuple[torch.Tensor, int]:
    ids = torch.load(path, map_location="cpu")
    if ids.ndim == 1:
        if sequence_length <= 0:
            raise ValueError("--sequence_length is required when loading a one-dimensional token tensor.")
        usable = ids.numel() // sequence_length * sequence_length
        input_ids = ids[:usable].reshape(-1, sequence_length)
    elif ids.ndim == 2:
        input_ids = ids.to(dtype=torch.long, device="cpu")
        if sequence_length > 0 and sequence_length != input_ids.size(1):
            flat = input_ids.reshape(-1)
            usable = flat.numel() // sequence_length * sequence_length
            input_ids = flat[:usable].reshape(-1, sequence_length)
    else:
        raise ValueError(f"Expected a 1D or 2D token tensor, got shape {tuple(ids.shape)}")

    if max_sequences > 0:
        input_ids = input_ids[:max_sequences]
    if input_ids.numel() == 0:
        raise ValueError(f"No usable token ids were loaded from {path}.")
    return input_ids, int(input_ids.numel())


if __name__ == "__main__":
    main()
