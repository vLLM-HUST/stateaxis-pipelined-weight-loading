# stateaxis-pipelined-weight-loading

Extension ID: `org.vllm-hust.stateaxis-pipelined-weight-loading`

Identity-bound staged model loading with explicit cold/warm qualification.

This repository is the independent MOD boundary for StateAxis issues [#15](https://github.com/Qixin-Gaoke/stateaxis/issues/15).
It is deliberately `import_only`, default-off, and cannot be enabled. The split does
not inherit correctness, device, performance, or publication qualification from the
aggregate StateAxis repository.

## Evidence boundary

Status: **scoped-positive**.

Cold readiness improved in the frozen Qwen and EP8 cells without a measured warm/TPOT regression; activation remains artifact-bound.

The copied evidence and its SHA-256 are recorded in `PROVENANCE.json`. Negative,
failed, and inconclusive results are retained. Microbenchmarks and component results
must not be restated as online end-to-end gains.

## Install and inspect

```bash
python -m pip install .
vllm-hust-ext extension inspect org.vllm-hust.stateaxis-pipelined-weight-loading
vllm-hust-ext extension check org.vllm-hust.stateaxis-pipelined-weight-loading
```

Discovery does not enable the MOD. A future active revision must extract an
independently reviewable implementation, declare exclusive resources where needed,
and pass exactness, lifecycle, release, failure-recovery, and matched real-online
gates.

## Validate

```bash
python -m pip install -e '.[test]'
pytest -q
```

Maintainer: Shuhao Zhang (Tony), directly responsible; no advisor is declared.
