#!/usr/bin/env python3
"""Normalize an Ascend runtime-layout shard for generic sharded-state loading."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

MARKER_NAME = "stateaxis-sharded-state.json"
JOURNAL_NAME = ".stateaxis-sharded-state-normalization.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    marker = artifact / MARKER_NAME
    journal_path = artifact / JOURNAL_NAME
    if marker.exists():
        raise SystemExit(f"artifact is already normalized: {marker}")

    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal["source_manifest_sha256"] != args.source_manifest_sha256:
            raise SystemExit(
                "normalization journal belongs to a different source manifest"
            )
    else:
        journal = {
            "schema_version": "stateaxis-sharded-state-normalization/v1",
            "source_layout": "vllm-ascend-runtime-postprocess",
            "target_layout": "vllm-generic-preprocess",
            "source_manifest_sha256": args.source_manifest_sha256,
            "completed_files": [],
            "transformed_tensors": [],
        }
        write_json_atomic(journal_path, journal)

    completed = set(journal["completed_files"])
    for shard in sorted(artifact.glob("model-rank-*-part-*.safetensors")):
        if shard.name in completed:
            continue

        with safe_open(shard, framework="pt") as handle:
            transform_keys = [
                key
                for key in handle.keys()  # noqa: SIM118 - safe_open is not iterable
                if key.endswith(("w13_weight", "w2_weight"))
                and len(handle.get_slice(key).get_shape()) == 3
            ]

        if transform_keys:
            required = shard.stat().st_size + 1024**3
            available = shutil.disk_usage(artifact).free
            if available < required:
                raise OSError(
                    f"insufficient free space for atomic rewrite of {shard.name}: "
                    f"need {required}, have {available}"
                )

            tensors = load_file(shard)
            for key in transform_keys:
                before_shape = list(tensors[key].shape)
                tensors[key] = tensors[key].transpose(1, 2).contiguous()
                journal["transformed_tensors"].append(
                    {
                        "file": shard.name,
                        "key": key,
                        "before_shape": before_shape,
                        "after_shape": list(tensors[key].shape),
                    }
                )

            temporary = artifact / f".{shard.name}.normalizing"
            try:
                save_file(tensors, temporary)
                os.replace(temporary, shard)
            finally:
                temporary.unlink(missing_ok=True)
            # ``load_file`` tensors retain mmap-backed storage.  Release the
            # replaced inode before checking space for the next atomic rewrite.
            del tensors
            gc.collect()

        journal["completed_files"].append(shard.name)
        journal["last_completed_size"] = shard.stat().st_size
        write_json_atomic(journal_path, journal)

    marker_payload = {
        **journal,
        "artifact_layout": "vllm-generic-preprocess",
        "normalization_complete": True,
    }
    write_json_atomic(marker, marker_payload)
    journal_path.unlink()


if __name__ == "__main__":
    main()
