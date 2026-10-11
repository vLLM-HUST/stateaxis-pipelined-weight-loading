import hashlib
import json
from pathlib import Path

from vllm_hust_ext.manifest import activation_blocker, load_manifest

import stateaxis_pipelined_weight_loading

ROOT = Path(__file__).resolve().parents[1]


def test_active_manifest_is_hash_bound() -> None:
    path = Path(stateaxis_pipelined_weight_loading.__file__).with_name(
        "vllm-hust-extension-v0.3.json"
    )
    manifest = load_manifest(path)
    assert manifest.bundle_id == stateaxis_pipelined_weight_loading.MOD_ID
    assert manifest.schema_version == "0.3-experimental"
    assert activation_blocker(manifest) is None
    raw = json.loads(path.read_text())
    digest = hashlib.sha256((ROOT / "RESEARCH_MANIFEST.json").read_bytes()).hexdigest()
    binding = raw["activation"]["additional_config"]["stateaxis_mod"]
    assert binding["manifest_sha256"] == digest


def test_contract_is_complete_and_fail_closed() -> None:
    config = stateaxis_pipelined_weight_loading.pipelined_weight_loading()
    assert config.enabled and config.qwen2_only
    assert config.candidate_policy == "pinned_double_buffer_async_v1"
    assert config.serial_default_when_disabled
    assert config.staging_slots == 2 and config.chunk_bytes == 64 * 1024 * 1024
    assert config.incremental_full_byte_sha256
    assert config.event_guarded_slot_reuse
    assert config.execution_artifact_plan_digest_bound
    assert config.packed_weights_root_bound
    assert config.transactional_ready and config.fail_closed


def test_manifest_owns_exclusive_loader_resources() -> None:
    raw = json.loads(
        Path(stateaxis_pipelined_weight_loading.__file__)
        .with_name("vllm-hust-extension-v0.3.json")
        .read_text()
    )
    resources = {claim["resource"] for claim in raw["resource_claims"]}
    assert resources == {
        "stateaxis.worker.weight-loader",
        "stateaxis.device.weight-residency",
        "stateaxis.host.pinned-staging",
    }
