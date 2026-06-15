#!/usr/bin/env python3
"""Build a checklist for batch model derivation.

This helper is intentionally read-only. It does not edit deployment docs.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass


GPU_MEM_GB = {
    "K100_AI": 64,
    "BW1000": 64,
    "BW1100": 144,
}

QUANT_BYTES = {
    "BF16": 2.0,
    "FP8": 1.0,
    "FP8 W8A8": 1.0,
    "INT8": 1.0,
    "INT8 W8A8": 1.0,
    "W8A8": 1.0,
    "FP4": 0.5,
    "INT4": 0.5,
    "INT4 W4A16": 0.5,
    "INT4 W4A8": 0.5,
    "AWQ": 0.5,
}


@dataclass
class TargetPlan:
    model_id: str
    is_full_id: bool
    params_b: float | None
    quantization: str
    estimated_weight_gb: float | None


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def is_full_model_id(model_id: str) -> bool:
    return "/" in model_id and not model_id.startswith("/") and not model_id.endswith("/")


def extract_params_b(model_id: str) -> float | None:
    # Prefer the largest explicit parameter size in names such as 35B-A3B.
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*[bB]", model_id)
    if not matches:
        return None
    return max(float(match) for match in matches)


def allowed_tp(value: float) -> int:
    if value <= 1:
        return 1
    if value <= 8:
        return 2 ** math.ceil(math.log2(value))
    tp = 8
    while tp < value:
        tp *= 2
    return tp


def estimate_tp(weight_gb: float, gpu_mem_gb: int, reserve: float) -> int:
    usable_mem = gpu_mem_gb * (1.0 - reserve)
    return allowed_tp(weight_gb / usable_mem)


def normalize_quantization(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().upper())


def build_plan(args: argparse.Namespace) -> list[TargetPlan]:
    quant = normalize_quantization(args.quantization)
    multiplier = QUANT_BYTES.get(quant)
    plans: list[TargetPlan] = []
    for target in split_csv(args.targets):
        params_b = extract_params_b(target)
        weight_gb = params_b * multiplier if params_b is not None and multiplier is not None else None
        plans.append(
            TargetPlan(
                model_id=target,
                is_full_id=is_full_model_id(target),
                params_b=params_b,
                quantization=quant,
                estimated_weight_gb=weight_gb,
            )
        )
    return plans


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a read-only batch add-model checklist.")
    parser.add_argument("--reference", required=True, help="Reference model ID or exact model name.")
    parser.add_argument("--targets", required=True, help="Comma-separated target model IDs.")
    parser.add_argument("--framework", required=True, choices=["vLLM", "SGLang", "vllm", "sglang"])
    parser.add_argument("--version", required=True, help="Framework version, e.g. 0.18.")
    parser.add_argument("--hardware", required=True, help="Comma-separated hardware: BW1100,BW1000,K100_AI.")
    parser.add_argument("--quantization", default="BF16", help="BF16, FP8 W8A8, INT8 W8A8, INT4 W4A8, AWQ, etc.")
    parser.add_argument("--reserve", type=float, default=0.30, help="GPU memory reserve ratio. Default: 0.30.")
    args = parser.parse_args()

    framework = "vLLM" if args.framework.lower() == "vllm" else "SGLang"
    hardware = split_csv(args.hardware)
    unknown_hw = [item for item in hardware if item not in GPU_MEM_GB]
    if unknown_hw:
        raise SystemExit(f"Unsupported hardware: {', '.join(unknown_hw)}")

    plans = build_plan(args)

    print("# Batch add-model plan")
    print()
    print(f"- Reference: {args.reference}")
    print(f"- Framework: {framework} {args.version}")
    print(f"- Hardware: {', '.join(hardware)}")
    print(f"- Quantization: {normalize_quantization(args.quantization)}")
    print(f"- Memory reserve: {args.reserve:.0%}")
    print()
    print("## Targets")
    for plan in plans:
        print()
        print(f"### {plan.model_id}")
        if not plan.is_full_id:
            print("- BLOCKED: target is not a full model ID. Ask the user for the exact ModelScope/HuggingFace ID.")
        if plan.params_b is None:
            print("- BLOCKED: cannot parse parameter size from model ID. Ask the user for model size/weight.")
        if plan.estimated_weight_gb is None:
            print("- BLOCKED: cannot estimate weight. Check quantization spelling or provide explicit weight.")
            continue
        print(f"- Parsed params: {plan.params_b:g}B")
        print(f"- Estimated weight: {plan.estimated_weight_gb:g} GB")
        for hw in hardware:
            tp = estimate_tp(plan.estimated_weight_gb, GPU_MEM_GB[hw], args.reserve)
            print(f"- {hw}: TP/card count {tp}")
        print("- Use add-model to insert table row(s), matching command section(s), anchors, and README matrix entry.")
        print("- Replace reference model ID/name with this target only after exact ID is confirmed.")

    print()
    print("## Required add-model handoff")
    print("- Load ../add-model/SKILL.md before editing.")
    print("- Use same framework/version/hardware/mode reference commands only.")
    print("- Recompute TP; do not copy TP from the reference model.")
    print("- Keep unrelated existing rows and command sections unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
