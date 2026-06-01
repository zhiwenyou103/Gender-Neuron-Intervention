"""Identify gender-specific MLP neurons from activation statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from gender_neuron_utils import activation_file_tag, infer_label_from_activation_path, model_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select gender-specific neurons from activation.*.train.<tag> files."
    )
    parser.add_argument("-m", "--model", default="meta-llama/Llama-2-7b-hf", help="Hugging Face model name or path.")
    parser.add_argument("--data_dir", default="processed_data", help="Directory containing activation statistics.")
    parser.add_argument("--output_dir", default="activation_mask", help="Directory for selected neuron masks.")
    parser.add_argument(
        "--model_tag",
        default="",
        help="Optional activation-file suffix override. Defaults to llama, bloom, or sanitized model key for Qwen.",
    )
    parser.add_argument("--method", choices=["combined", "ratio"], default="combined", help="Neuron selection method.")
    parser.add_argument("--device", default="auto", help="Device for tensor computations: auto, cpu, cuda, or cuda:<idx>.")
    parser.add_argument("--output_suffix", default="", help="Optional suffix for mask and statistics files.")

    parser.add_argument("--exclusivity_ratio", type=float, default=1.10, help="Ratio threshold for --method ratio.")
    parser.add_argument("--min_activation_threshold", type=float, default=0.02, help="Positive-rate threshold for --method ratio.")
    parser.add_argument("--top_rate", type=float, default=0.15, help="Maximum fraction of total neurons selected per gender for --method ratio.")

    parser.add_argument("--target_percent", type=float, default=0.01, help="Target fraction of total neurons selected per gender.")
    parser.add_argument("--per_layer_max_percent", type=float, default=0.020, help="Per-layer fraction cap for each gender.")
    parser.add_argument("--min_posrate", type=float, default=0.08, help="Minimum positive-activation rate for combined scoring.")
    parser.add_argument("--min_effect_size", type=float, default=0.5, help="Minimum normalized effect-size component.")
    parser.add_argument("--min_log_odds", type=float, default=0.7, help="Minimum normalized log-odds component.")
    parser.add_argument("--exclusivity_margin", type=float, default=0.4, help="Minimum best-minus-second score margin.")
    parser.add_argument("--adaptive_relax", action="store_true", help="Relax combined-method thresholds if too few candidates are found.")
    parser.add_argument("--min_candidates_per_gender", type=int, default=1000, help="Candidate target used with --adaptive_relax.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    file_tag = args.model_tag or activation_file_tag(args.model)
    labels, n, over_zero, sum1, sum2 = load_activation_data(Path(args.data_dir), file_tag, device)

    num_layers, intermediate_size, gender_count = over_zero.shape
    total_neurons = num_layers * intermediate_size
    print(f"Loaded labels: {', '.join(labels)}")
    print(f"Architecture from tensors: {num_layers} layers x {intermediate_size} MLP neurons")
    print(f"Total candidate neurons: {total_neurons:,}")

    if gender_count < 2:
        raise ValueError("At least two gender/style labels are required to identify exclusive neurons.")

    if args.method == "combined" and (sum1 is None or sum2 is None):
        print("sum1/sum2 moments are missing; falling back to ratio selection.")
        args.method = "ratio"

    if args.method == "ratio":
        masks, stats = select_by_ratio(
            labels,
            n,
            over_zero,
            args.exclusivity_ratio,
            args.min_activation_threshold,
            args.top_rate,
        )
        metadata = {
            "method": "ratio",
            "exclusivity_ratio": args.exclusivity_ratio,
            "min_activation_threshold": args.min_activation_threshold,
            "top_rate": args.top_rate,
        }
    else:
        masks, stats, thresholds = select_by_combined_score(
            labels,
            n,
            over_zero,
            sum1,
            sum2,
            args,
        )
        metadata = {
            "method": "combined",
            "target_percent": args.target_percent,
            "per_layer_max_percent": args.per_layer_max_percent,
            "requested_thresholds": {
                "min_posrate": args.min_posrate,
                "min_effect_size": args.min_effect_size,
                "min_log_odds": args.min_log_odds,
                "exclusivity_margin": args.exclusivity_margin,
            },
            "final_thresholds": thresholds,
        }

    save_results(args, labels, masks, stats, metadata, total_neurons)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_activation_data(
    data_dir: Path,
    file_tag: str,
    device: torch.device,
) -> tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    files = sorted(data_dir.glob(f"activation.*.train.{file_tag}"))
    if not files and file_tag == "bloom":
        files = sorted(data_dir.glob("activation.*.train.bloom*"))
    if not files:
        raise FileNotFoundError(
            f"No activation files matched {data_dir / f'activation.*.train.{file_tag}'}. "
            "Run collect_activations.py for each label first."
        )

    labeled_files = []
    for path in files:
        label = infer_label_from_activation_path(path, file_tag)
        if label is None and file_tag == "bloom":
            label = path.name.split(".")[1]
        if label is not None:
            labeled_files.append((label, path))
    labeled_files.sort(key=lambda item: item[0])

    labels: List[str] = []
    n_values: List[int] = []
    over_zero_values: List[torch.Tensor] = []
    sum1_values: List[torch.Tensor] = []
    sum2_values: List[torch.Tensor] = []

    for label, path in labeled_files:
        data = torch.load(path, map_location="cpu")
        required = {"n", "over_zero"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"{path} is missing required field(s): {sorted(missing)}")

        labels.append(label)
        n_values.append(int(data["n"]))
        over_zero_values.append(data["over_zero"].to(device=device, dtype=torch.float32))
        if "sum1" in data and "sum2" in data:
            sum1_values.append(data["sum1"].to(device=device, dtype=torch.float32))
            sum2_values.append(data["sum2"].to(device=device, dtype=torch.float32))
        print(f"Loaded {label}: {path.name}")

    n = torch.tensor(n_values, dtype=torch.float32, device=device)
    over_zero = torch.stack(over_zero_values, dim=-1)
    has_moments = len(sum1_values) == len(labels)
    sum1 = torch.stack(sum1_values, dim=-1) if has_moments else None
    sum2 = torch.stack(sum2_values, dim=-1) if has_moments else None
    return labels, n, over_zero, sum1, sum2


def select_by_ratio(
    labels: List[str],
    n: torch.Tensor,
    over_zero: torch.Tensor,
    exclusivity_ratio: float,
    min_activation_threshold: float,
    top_rate: float,
) -> Tuple[List[List[torch.Tensor]], List[Dict[str, float]]]:
    num_layers, intermediate_size, gender_count = over_zero.shape
    total_neurons = num_layers * intermediate_size
    activation_probs = over_zero / n.view(1, 1, -1).clamp_min(1)
    sorted_probs, sorted_indices = torch.sort(activation_probs, dim=-1, descending=True)
    highest = sorted_probs[:, :, 0]
    second = sorted_probs[:, :, 1]
    dominant = sorted_indices[:, :, 0]
    exclusivity = highest / (second + 1e-8)
    eligible = (highest >= min_activation_threshold) & (exclusivity >= exclusivity_ratio)

    masks: List[List[torch.Tensor]] = []
    stats: List[Dict[str, float]] = []
    for gender_idx, label in enumerate(labels):
        gender_mask = eligible & (dominant == gender_idx)
        scores = torch.where(gender_mask, exclusivity, torch.zeros_like(exclusivity))
        total_exclusive = int(gender_mask.sum().item())
        select_count = min(total_exclusive, int(top_rate * total_neurons))
        layer_masks = empty_layer_masks(num_layers)

        if select_count > 0:
            _, top_indices = torch.topk(scores.flatten(), select_count)
            layer_indices = torch.div(top_indices, intermediate_size, rounding_mode="floor")
            neuron_indices = top_indices % intermediate_size
            layer_masks = group_flat_indices(layer_indices, neuron_indices, num_layers)

        selected_scores = collect_selected_values(exclusivity, layer_masks)
        masks.append(layer_masks)
        stats.append(
            {
                "gender": label,
                "total_exclusive": total_exclusive,
                "selected": sum(len(layer) for layer in layer_masks),
                "avg_score": float(np.mean(selected_scores)) if selected_scores else 0.0,
            }
        )
        print_selection_summary(label, stats[-1], "exclusivity ratio")

    return masks, stats


def select_by_combined_score(
    labels: List[str],
    n: torch.Tensor,
    over_zero: torch.Tensor,
    sum1: torch.Tensor,
    sum2: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[List[List[torch.Tensor]], List[Dict[str, float]], Dict[str, float]]:
    scores = compute_combined_scores(n, over_zero, sum1, sum2)
    primary = scores[:, :, :, 0]
    effect = scores[:, :, :, 1]
    log_odds = scores[:, :, :, 2]
    pos_rate = scores[:, :, :, 4]
    num_layers, intermediate_size, gender_count = over_zero.shape
    total_neurons = num_layers * intermediate_size

    min_pos = args.min_posrate
    min_effect = args.min_effect_size
    min_log_odds = args.min_log_odds
    margin = args.exclusivity_margin

    best_score, best_gender, margin_vec = rank_gender_scores(
        primary, effect, log_odds, pos_rate, min_pos, min_effect, min_log_odds
    )

    if args.adaptive_relax:
        for iteration in range(20):
            counts = [
                int(((best_gender == g) & (best_score > 0) & (margin_vec >= margin)).sum().item())
                for g in range(gender_count)
            ]
            print(f"Adaptive relaxation {iteration}: {dict(zip(labels, counts))}")
            if all(count >= args.min_candidates_per_gender for count in counts):
                break
            min_pos = max(0.01, min_pos * 0.9)
            min_effect = max(0.1, min_effect * 0.9)
            min_log_odds = max(0.1, min_log_odds * 0.9)
            margin = max(0.1, margin * 0.9)
            best_score, best_gender, margin_vec = rank_gender_scores(
                primary, effect, log_odds, pos_rate, min_pos, min_effect, min_log_odds
            )

    target_per_gender = max(1, int(args.target_percent * total_neurons))
    per_layer_cap = max(1, int(args.per_layer_max_percent * intermediate_size))
    layer_ids = torch.div(
        torch.arange(total_neurons, device=best_score.device),
        intermediate_size,
        rounding_mode="floor",
    )
    neuron_ids = torch.arange(total_neurons, device=best_score.device) % intermediate_size

    masks: List[List[torch.Tensor]] = []
    stats: List[Dict[str, float]] = []
    for gender_idx, label in enumerate(labels):
        candidate_mask = (best_gender == gender_idx) & (best_score > 0) & (margin_vec >= margin)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        layer_masks = empty_layer_masks(num_layers)
        selected_scores: List[float] = []

        if candidate_indices.numel() > 0:
            candidate_scores = best_score[candidate_indices]
            _, order = torch.sort(candidate_scores, descending=True)
            per_layer_counts = torch.zeros(num_layers, dtype=torch.int64, device=best_score.device)

            for flat_idx in candidate_indices[order]:
                if sum(len(layer) for layer in layer_masks) >= target_per_gender:
                    break
                layer_idx = int(layer_ids[flat_idx].item())
                if per_layer_counts[layer_idx] >= per_layer_cap:
                    continue
                neuron_idx = int(neuron_ids[flat_idx].item())
                layer_masks[layer_idx] = torch.cat(
                    [layer_masks[layer_idx], torch.tensor([neuron_idx], dtype=torch.long)]
                )
                per_layer_counts[layer_idx] += 1
                selected_scores.append(float(best_score[flat_idx].item()))

        masks.append(layer_masks)
        stats.append(
            {
                "gender": label,
                "total_exclusive": int(candidate_indices.numel()),
                "selected": sum(len(layer) for layer in layer_masks),
                "avg_score": float(np.mean(selected_scores)) if selected_scores else 0.0,
            }
        )
        print_selection_summary(label, stats[-1], "specificity score")

    thresholds = {
        "min_posrate": float(min_pos),
        "min_effect_size": float(min_effect),
        "min_log_odds": float(min_log_odds),
        "exclusivity_margin": float(margin),
    }
    return masks, stats, thresholds


def compute_combined_scores(
    n: torch.Tensor,
    over_zero: torch.Tensor,
    sum1: torch.Tensor,
    sum2: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-4
    n_broadcast = n.view(1, 1, -1).clamp_min(1)
    mean = sum1 / n_broadcast
    variance = torch.clamp(sum2 / n_broadcast - mean.square(), min=eps)
    positive_rate = torch.clamp(over_zero / n_broadcast, eps, 1 - eps)
    gender_count = n.numel()
    total_n = n.sum()

    components = []
    for gender_idx in range(gender_count):
        other = torch.ones(gender_count, dtype=torch.bool, device=n.device)
        other[gender_idx] = False
        n_rest = (total_n - n[gender_idx]).clamp_min(1)
        rest_sum1 = sum1[:, :, other].sum(dim=-1)
        rest_sum2 = sum2[:, :, other].sum(dim=-1)
        rest_mean = rest_sum1 / n_rest
        rest_variance = torch.clamp(rest_sum2 / n_rest - rest_mean.square(), min=eps)

        target_mean = mean[:, :, gender_idx]
        target_variance = variance[:, :, gender_idx]
        target_rate = positive_rate[:, :, gender_idx]
        rest_rate = torch.clamp(over_zero[:, :, other].sum(dim=-1) / n_rest, eps, 1 - eps)

        effect = torch.relu((target_mean - rest_mean) / torch.sqrt(0.5 * (target_variance + rest_variance) + eps))
        log_odds = torch.relu(_logit(target_rate) - _logit(rest_rate))
        relative_mean = torch.relu((target_mean - rest_mean) / (torch.abs(rest_mean) + 1e-2))

        effect = torch.nan_to_num(effect)
        log_odds = torch.nan_to_num(log_odds)
        relative_mean = torch.nan_to_num(relative_mean)
        score = 0.45 * effect + 0.35 * log_odds + 0.20 * relative_mean
        components.append(torch.stack([score, effect, log_odds, relative_mean, target_rate], dim=0))

    scores = torch.stack(components, dim=0).permute(0, 2, 3, 1)
    for component_idx in range(1, 4):
        component = scores[:, :, :, component_idx]
        mean_value = component.mean(dim=(1, 2), keepdim=True)
        std_value = component.std(dim=(1, 2), keepdim=True).clamp_min(eps)
        scores[:, :, :, component_idx] = (component - mean_value) / std_value
    scores[:, :, :, 0] = 0.45 * scores[:, :, :, 1] + 0.35 * scores[:, :, :, 2] + 0.20 * scores[:, :, :, 3]
    return scores


def rank_gender_scores(
    primary: torch.Tensor,
    effect: torch.Tensor,
    log_odds: torch.Tensor,
    pos_rate: torch.Tensor,
    min_pos: float,
    min_effect: float,
    min_log_odds: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = (pos_rate >= min_pos) & (effect >= min_effect) & (log_odds >= min_log_odds)
    filtered = torch.where(keep, primary, torch.zeros_like(primary))
    flat = filtered.reshape(filtered.size(0), -1)
    sorted_values, sorted_indices = torch.sort(flat, dim=0, descending=True)
    best_score = sorted_values[0]
    second_score = sorted_values[1]
    best_gender = sorted_indices[0]
    return best_score, best_gender, best_score - second_score


def group_flat_indices(
    layer_indices: torch.Tensor,
    neuron_indices: torch.Tensor,
    num_layers: int,
) -> List[torch.Tensor]:
    grouped = empty_layer_masks(num_layers)
    for layer_idx, neuron_idx in zip(layer_indices.cpu(), neuron_indices.cpu()):
        layer = int(layer_idx.item())
        grouped[layer] = torch.cat([grouped[layer], torch.tensor([int(neuron_idx.item())], dtype=torch.long)])
    return grouped


def collect_selected_values(values: torch.Tensor, masks: List[torch.Tensor]) -> List[float]:
    selected = []
    for layer_idx, layer_mask in enumerate(masks):
        for neuron_idx in layer_mask:
            selected.append(float(values[layer_idx, int(neuron_idx.item())].item()))
    return selected


def empty_layer_masks(num_layers: int) -> List[torch.Tensor]:
    return [torch.tensor([], dtype=torch.long) for _ in range(num_layers)]


def save_results(
    args: argparse.Namespace,
    labels: List[str],
    masks: List[List[torch.Tensor]],
    stats: List[Dict[str, float]],
    metadata: Dict,
    total_neurons: int,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix
    output_prefix = f"{model_key(args.model)}_exclusive{suffix}"
    mask_path = output_dir / output_prefix
    torch.save(masks, mask_path)

    gender_map_path = output_dir / f"gender_map{suffix}.txt"
    gender_map_path.write_text(
        "".join(f"{idx}: {label}\n" for idx, label in enumerate(labels)),
        encoding="utf-8",
    )

    summary = {
        "model": args.model,
        "total_neurons": total_neurons,
        "total_selected": int(sum(item["selected"] for item in stats)),
        "labels": stats,
        **metadata,
    }
    json_path = output_dir / f"{output_prefix}_stats.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    text_path = output_dir / f"{output_prefix}_stats.txt"
    text_path.write_text(format_text_summary(summary), encoding="utf-8")

    print(f"Saved masks to {mask_path}")
    print(f"Saved label mapping to {gender_map_path}")
    print(f"Saved statistics to {json_path} and {text_path}")


def format_text_summary(summary: Dict) -> str:
    lines = [
        "Gender-Specific Neuron Analysis",
        "=" * 36,
        f"Model: {summary['model']}",
        f"Method: {summary['method']}",
        f"Total selected neurons: {summary['total_selected']:,}",
        f"Total candidate neurons: {summary['total_neurons']:,}",
        "",
    ]
    for item in summary["labels"]:
        lines.extend(
            [
                f"Gender/style '{item['gender']}':",
                f"  candidates: {item['total_exclusive']:,}",
                f"  selected: {item['selected']:,}",
                f"  average score: {item['avg_score']:.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def print_selection_summary(label: str, stats: Dict[str, float], score_name: str) -> None:
    print(f"{label}: {stats['selected']:,} selected from {stats['total_exclusive']:,} candidates")
    if stats["selected"]:
        print(f"  average {score_name}: {stats['avg_score']:.4f}")


def _logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.log(probability) - torch.log1p(-probability)


if __name__ == "__main__":
    main()
