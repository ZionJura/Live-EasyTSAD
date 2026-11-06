PHASE: EVIDENCE — build the reproduction contract for `{method}`.

Paper: {title} ({paper_ref})
Official code: {code_info}

Do:
1. Read the paper (`{task_dir}/paper/paper.md`; method, experiments, appendix/
   implementation details) and the official code if present (model, training loop,
   anomaly score, default arguments / run scripts for the benchmark datasets).
2. Call `mcp__tsad__save_contract` with the contract JSON (schema below). Every
   claim gets `status`: "explicit" (stated in the paper), "inferred" (from the
   official code or unambiguous from context; cite file:line) or "unspecified"
   (neither; give the neutral default you will use). `source` must locate the evidence
   ("Sec. 4.1", "Table 6", "code: solver.py:L120"). Add `config_key` when the value
   will live in config.toml `[Model_Params.Default]` (or `Data_Params.preprocess`).
   Only scalar hyper-parameters (numbers, booleans, short identifiers) get a
   `config_key`; descriptive claims (formulas, policies) never do.
   Required claim ids: data.preprocess, data.window_size, model.architecture,
   train.optimizer, train.lr, train.epochs, train.batch_size, score.function,
   threshold.protocol. Add any other hyper-parameter the algorithm needs.
3. Fix validation errors returned by the tool until it accepts the contract.
4. Do NOT write the method yet.

Contract schema:
```json
{contract_example}
```
