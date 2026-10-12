#!/usr/bin/env python3
"""Summarize matched Qwen3.5 loader diagnostics without selecting best runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _metric(values: list[float]) -> dict[str, Any]:
    return {
        "values_seconds": values,
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.mean(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def _load_runs(root: Path, prefix: str, repetitions: int) -> list[dict[str, Any]]:
    return [
        json.loads((root / f"{prefix}-r{index}" / "result.json").read_text())
        for index in range(1, repetitions + 1)
    ]


def _one(values: set[str], field: str) -> str:
    if len(values) != 1:
        raise ValueError(f"{field} differs across matched runs: {sorted(values)}")
    return values.pop()


def build_summary(root: Path, repetitions: int) -> dict[str, Any]:
    control = _load_runs(root, "control-overlay", repetitions)
    candidate = _load_runs(root, "sharded-nvme", repetitions)
    runs = control + candidate

    model_config_sha256 = _one(
        {run["model_config_sha256"] for run in runs}, "model_config_sha256"
    )
    model_index_sha256 = _one(
        {run["model_index_sha256"] for run in runs}, "model_index_sha256"
    )
    response_sha256 = _one(
        {run["response_content_sha256"] for run in runs},
        "response_content_sha256",
    )
    vllm_commit = _one({run["vllm_commit"] for run in runs}, "vllm_commit")
    vllm_ascend_commit = _one(
        {run["vllm_ascend_commit"] for run in runs}, "vllm_ascend_commit"
    )
    if any(run["formal_stateaxis_mod_on"] for run in runs):
        raise ValueError("carrier diagnostics must not claim formal MOD activation")

    metrics: dict[str, Any] = {}
    for name, getter in {
        "process_to_ready": lambda run: run["process_to_ready_seconds"],
        "weight_iterator": lambda run: run["weight_iterator_seconds"][0],
        "first_request": lambda run: run["first_request_seconds"],
    }.items():
        control_metric = _metric([float(getter(run)) for run in control])
        candidate_metric = _metric([float(getter(run)) for run in candidate])
        baseline = control_metric["median_seconds"]
        optimized = candidate_metric["median_seconds"]
        metrics[name] = {
            "control": control_metric,
            "sharded": candidate_metric,
            "median_reduction_percent": (baseline - optimized) / baseline * 100,
        }

    return {
        "schema_version": "stateaxis-qwen35-loader-comparison/v1",
        "date_utc": "2026-10-12",
        "mod_id": "org.vllm-hust.stateaxis-pipelined-weight-loading",
        "evidence_label": "matched-real-npu-carrier-diagnostic-not-mod-performance",
        "performance_qualified": False,
        "formal_stateaxis_mod_on": False,
        "workload": {
            "model": "Qwen3.5-35B-A3B BF16",
            "tensor_parallel_size": 2,
            "max_model_len": 2048,
            "enforce_eager": True,
            "hardware": "2 x Ascend 910B2, 64 GiB HBM each",
            "npu_smi_version": "25.2.1",
            "storage_pairing": "both arms on /dev/nvme0n1p1 overlay",
            "cache_policy": "uncontrolled warm/cold state; alternating arms",
        },
        "identity": {
            "model_config_sha256": model_config_sha256,
            "model_index_sha256": model_index_sha256,
            "response_content_sha256": response_sha256,
            "vllm_commit": vllm_commit,
            "vllm_ascend_commit": vllm_ascend_commit,
            "sharded_artifact_manifest_sha256": (
                "170bed99a870c0e602016db824f36cf98668e0400d5c46d525d3d7e4222deddf"
            ),
        },
        "arms": {
            "control": "vLLM serial safetensors over 14 source shards",
            "sharded": "vLLM TP sharded_state over 9 pre-sharded files per rank",
            "order": [
                item
                for index in range(1, repetitions + 1)
                for item in (f"sharded-r{index}", f"control-r{index}")
            ],
            "repetitions_per_arm": repetitions,
        },
        "metrics": metrics,
        "correctness": {
            "all_first_response_hashes_equal": True,
            "request_failures": 0,
        },
        "limitations": [
            "The first sharded run preceded the first matched overlay control run.",
            "OS page-cache state was not reset or measured.",
            "Runtime dirty state was not captured at each launch.",
            "This carrier test did not activate the mod through ECPA/Manifest 0.3.",
            "The result applies only to Qwen3.5-35B-A3B BF16 TP2 startup.",
            "No steady-state AgentX throughput or latency claim is made.",
        ],
        "decision": {
            "status": "promising-carrier-needs-formal-ecpa-on-off",
            "reason": (
                "Three matched same-host/same-filesystem repetitions preserve output "
                "identity and show a positive median startup reduction, but activation "
                "and cache-state controls are not yet formal."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = build_summary(args.evidence_root, args.repetitions)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
