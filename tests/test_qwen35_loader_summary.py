import json
from pathlib import Path

import pytest

from tools.summarize_qwen35_loader import build_summary


def _write_run(root: Path, name: str, ready: float, iterator: float) -> None:
    directory = root / name
    directory.mkdir()
    (directory / "result.json").write_text(
        json.dumps(
            {
                "formal_stateaxis_mod_on": False,
                "model_config_sha256": "config",
                "model_index_sha256": "index",
                "response_content_sha256": "response",
                "vllm_commit": "vllm",
                "vllm_ascend_commit": "ascend",
                "process_to_ready_seconds": ready,
                "weight_iterator_seconds": [iterator],
                "first_request_seconds": 1.0,
            }
        )
    )


def test_summary_uses_all_repetitions_and_medians(tmp_path: Path) -> None:
    for index, value in enumerate((30.0, 20.0, 10.0), start=1):
        _write_run(tmp_path, f"control-overlay-r{index}", value, value / 2)
    for index, value in enumerate((15.0, 10.0, 5.0), start=1):
        _write_run(tmp_path, f"sharded-nvme-r{index}", value, value / 2)

    result = build_summary(tmp_path, 3)

    assert result["arms"]["repetitions_per_arm"] == 3
    assert result["metrics"]["process_to_ready"]["control"]["median_seconds"] == 20.0
    assert result["metrics"]["process_to_ready"]["sharded"]["median_seconds"] == 10.0
    assert result["metrics"]["process_to_ready"][
        "median_reduction_percent"
    ] == pytest.approx(50.0)


def test_summary_rejects_mismatched_outputs(tmp_path: Path) -> None:
    for arm in ("control-overlay", "sharded-nvme"):
        _write_run(tmp_path, f"{arm}-r1", 1.0, 1.0)
    payload_path = tmp_path / "sharded-nvme-r1" / "result.json"
    payload = json.loads(payload_path.read_text())
    payload["response_content_sha256"] = "different"
    payload_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="response_content_sha256 differs"):
        build_summary(tmp_path, 1)
