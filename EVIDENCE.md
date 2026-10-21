# Canonical evidence

Raw measured data lives in vllm-hust-benchmark, not this MOD's Git tree.
This is a custody migration, not a new experiment or qualification.

- [Original report and limitations](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/README.md)
- [Original detailed evidence index](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/EVIDENCE.md)
- [Original machine-readable claims](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/MOD_METADATA.json)
- [Complete frozen snapshot](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/)
- [Byte inventory and source identity](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/manifest.json)

EVIDENCE_SOURCES.json pins the benchmark commit and manifest SHA256. The helper
checks every downloaded file against its original length and SHA256, including
empty or failed records. It never executes downloaded evidence. Use a cache
outside this repository:

```bash
python tools/benchmark_evidence.py --all --destination /tmp/stateaxis-pipelined-weight-loading-evidence
```

Do not commit cache contents or new experiment outputs here. Submit evidence to
benchmark and update the pinned source index after integrity verification.
