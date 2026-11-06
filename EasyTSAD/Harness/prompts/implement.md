PHASE: IMPLEMENT — write the EasyTSAD method `{method}` from the approved contract
(`{task_dir}/contract.json`).

Do:
1. Choose the environment (reuse `torch2` if all dependencies exist there; else a new
   dedicated env via `scripts/tsad new` + `scripts/tsad add`). Register `{method}` in
   exactly one env in envs/registry.toml.
2. Write `{method_dir}/` following the system rules: `src/` algorithm (no EasyTSAD
   imports; adapt the official code where possible), the `{method}.py` adapter,
   `__init__.py`, `config.toml` whose values match the contract claims with a
   `config_key`.
3. Quick self-test inside the env (keep any test script in `{task_dir}/scratch/`, never in the method folder), e.g. `scripts/tsad run {method} -- python -c "..."`
   on a small random array through `src/` (CPU or a few seconds of GPU).
4. Call `mcp__tsad__check_method` until it reports ok.
5. Record every deviation / assumption with `mcp__tsad__record_note`.

Exit check run by the harness after your turn: method files present, class name,
registry entry, `src/` free of EasyTSAD imports, import check in the env, config
matches contract, file guard clean.
