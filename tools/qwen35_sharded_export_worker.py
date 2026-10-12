"""Worker used only while exporting a TP-sharded checkpoint.

The normal vLLM engine profiles a forward pass before it exposes the worker
RPC used by ``save_sharded_state_offline.py``.  Exporting does not execute
requests or need a production-sized KV cache, so that profile is unnecessary
and can require kernels which are unrelated to checkpoint serialization.
"""

from vllm_ascend.worker.worker import NPUWorker


class NPUShardedStateExportWorker(NPUWorker):
    """Skip the inference-only memory profile for an export process."""

    def determine_available_memory(self) -> int:
        # The engine requires a non-zero cache allocation before it accepts
        # RPCs.  One GiB is deliberately small and is never used for inference.
        return 1024**3
