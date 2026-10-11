# Evidence index

- Current activation contract: `evidence/ecpa-contract-20261011/RESULT.json`
  (`manager-contract-artifact-binding-non-performance`).
- Frozen real-NPU result: `evidence/ISSUE15_WEIGHT_LOADER_RESULT.md`
  (`matched-real-npu-qwen-cold-warm-scoped-positive`).

The Qwen result is a real matched cold/warm startup result with steady-serving
regression gates. The EP8 result validates only the loader, resource bounds and
transactional barrier. Current mock-worker checks validate activation and
identity propagation, not device performance.
