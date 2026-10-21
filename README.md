# stateaxis-pipelined-weight-loading

Extension ID: `org.vllm-hust.stateaxis-pipelined-weight-loading`

Experimental, default-off authority for the StateAxis Qwen2 startup loader.
Extension Manager/ECPA verifies the research digest. StateAxis then admits only
a canonical `pinned_double_buffer_async_v1` plan whose SHA-256, packed-weight
root, geometry and execution-artifact-v6 identity all agree. The validated
absolute plan path is injected only into the child worker; without ECPA
authority the host removes the ambient variable and the worker retains its
`serial_pageable_sync` default.

The candidate uses two 64 MiB pinned staging slots, incremental full-byte
SHA-256, asynchronous H2D, event-guarded slot reuse, per-pack verification and
transactional Ready. Any plan, artifact, root, geometry or worker-start failure
fails closed.

## Evidence boundary

Original measured figures, raw records, failed attempts and limitations are now
maintained in [benchmark's frozen report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/README.md) and
[original evidence index](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/EVIDENCE.md). See [EVIDENCE.md](EVIDENCE.md)
for immutable source links and verified external-cache retrieval.

Historical loader results remain workload-scoped. The newer sharded carrier
was not activated through ECPA and did not control cache state; it does not
establish a steady-serving or general performance claim.

```bash
python -m pip install -e '.[test]'
pytest -q
python tools/benchmark_evidence.py --destination /tmp/weight-loader-evidence \
  --path evidence/ecpa-contract-20261011/manager-config.json
VLLM_HUST_EXT_CONFIG=/tmp/weight-loader-evidence/evidence/ecpa-contract-20261011/manager-config.json \
  vllm-hust-ext extension check org.vllm-hust.stateaxis-pipelined-weight-loading
```

Maintainer: Shuhao Zhang (Tony), directly responsible; no advisor is declared.
