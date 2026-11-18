---
name: research-interface
description: Map official source inputs and anomaly scores to EasyTSAD in reproduction/adapter.yaml with source, target, align and evidence sections.
---

Inspect the method's `upstream/` entry points and EasyTSAD's actual BaseMethod/data interfaces. Read existing adapters only as interface examples; preserve the assigned official algorithm.

Write `reproduction/adapter.yaml` with these top-level sections:
- `source`: official entry point, callable signatures, fit/test inputs, preprocessing and score output shapes.
- `target`: EasyTSAD method class, supported training schema and adapter call/return requirements.
- `align`: exact mapping of arrays, channel/window dimensions, normalization fit scope, temporal positions, score direction and output length.
- `evidence`: repository paths/lines and paper pages supporting the mapping, plus unresolved differences.

Describe how every test position maps to an anomaly score, including any initial positions omitted by windowing. Keep thresholding and metric computation in the existing evaluator. Do not change labels, metrics, the hyperparameter artifact or the environment branch outputs. Return a concrete mapping for IMPLEMENT without starting training.
