PHASE: EVIDENCE — build the reproduction contract for `DIFR`.

Paper: Deep Isolation Forest for Anomaly Detection (2206.06602)
Official code: https://github.com/xuhongzuo/deep-iForest cloned at /home/v-zzr25/workplace/Live-EasyTSAD/EasyTSAD/Harness/tasks/DIFR/code/deep-iForest (commit e77692bbb39c851a580c37154c0a7d0c3a460bdf)

Do:
1. Read the paper (`/home/v-zzr25/workplace/Live-EasyTSAD/EasyTSAD/Harness/tasks/DIFR/paper/paper.md`; method, experiments, appendix/
   implementation details) and the official code if present (model, training loop,
   anomaly score, default arguments / run scripts for the benchmark datasets).
2. Call `mcp__tsad__save_contract` with the contract JSON (schema below). Every
   claim gets `status`: "explicit" (stated in the paper), "inferred" (from the
   official code or unambiguous from context; cite file:line) or "unspecified"
   (neither; give the neutral default you will use). `source` must locate the evidence
   ("Sec. 4.1", "Table 6", "code: solver.py:L120"). Add `config_key` when the value
   will live in config.toml `[Model_Params.Default]` (or `Data_Params.preprocess`).
   Required claim ids: data.preprocess, data.window_size, model.architecture,
   train.optimizer, train.lr, train.epochs, train.batch_size, score.function,
   threshold.protocol. Add any other hyper-parameter the algorithm needs.
3. Fix validation errors returned by the tool until it accepts the contract.
4. Do NOT write the method yet.

Contract schema:
```json
{
  "method": "Name",
  "paper": {
    "title": "...",
    "arxiv_id": "...",
    "venue": "...",
    "year": 2022
  },
  "task_type": "UTS|MTS|both",
  "summary": "two-sentence description of the algorithm",
  "claims": [
    {
      "id": "data.window_size",
      "value": 100,
      "status": "explicit|inferred|unspecified",
      "source": "Sec. 4.1 / Table 5 / code: solver.py:L42",
      "mutable": false,
      "config_key": "window_size"
    }
  ],
  "score": {
    "definition": "how the anomaly score is computed",
    "alignment": "how scores map to timestamps"
  },
  "dependencies": [
    "torch>=2",
    "..."
  ],
  "official_code": {
    "url": "...",
    "commit": "...",
    "used_files": [
      "model/AnomalyTransformer.py"
    ]
  },
  "open_questions": [
    "..."
  ]
}
```
