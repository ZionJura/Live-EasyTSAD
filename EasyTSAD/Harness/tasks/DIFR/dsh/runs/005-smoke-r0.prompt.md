PHASE: SMOKE — prove `DIFR` runs end-to-end through the real EasyTSAD /
LiveLeaderBoard pipeline on a small case (naive schema, NAB curves Twitter_volume_AAPL, Twitter_volume_AMZN, <= 2 epochs).

Do:
1. Call `mcp__tsad__run_smoke` (you have 8 smoke runs left in this task).
2. If it fails, read the returned `failure_class`, `error` and `log_tail`, fix the
   cause, and run it again. Allowed edit scope per failure class:
   - ENVIRONMENT: the method's own env (envs/<env>/pyproject.toml via `scripts/tsad add`).
   - ADAPTER_INTERFACE: `EasyTSAD/Methods/DIFR/DIFR.py` only (shapes, lifecycle, score alignment).
   - ALGORITHM_IMPLEMENTATION: `EasyTSAD/Methods/DIFR/src/` — keep it faithful to the contract;
     record any change of a contract value with `mcp__tsad__record_note` (kind="deviation").
   - RESOURCE_EXECUTION: runtime settings only (device, batch size for inference,
     num_workers); never change the algorithm; record it.
3. Stop when a smoke run succeeds with finite scores. Metric values are NOT a target.
