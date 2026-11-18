You integrate `{method}` into EasyTSAD so the existing evaluation pipeline can run it.

Repository: {root}
Method directory: {method_dir}/
Task inputs and execution logs: {task_dir}/

Use the official repository already copied into the method directory. Keep its
layout and reuse its implementation and default experiment configuration. Add
`{method}.py`, `__init__.py` and `config.toml` to connect it to EasyTSAD. Make the
runtime compatibility and bug fixes needed to execute the method. There is no
source-hash comparison, fidelity review or evidence-document deliverable here.
If no repository is supplied, implement from the supplied paper.

The adapter subclasses BaseMethod (`from .. import BaseMethod`,
`from ...DataFactory import TSData`) and provides `train_valid_phase`, `test_phase`
and `anomaly_score`. Return one-dimensional scores aligned with the test series;
EasyTSAD aligns shorter scores from the end. Wire the class in `__init__.py` and
map inputs through `[Data_Params]` and `[Model_Params.Default]` in `config.toml`.
Reuse existing method adapters when useful. Keep EasyTSAD's data loading, metrics,
thresholding, other methods and leaderboard implementation unchanged.

Use the method environment already prepared by the environment worker. Address
only runtime requirements exposed by the integration, including external programs
(such as Java), required files and their paths. Update the method-local dependency
project when necessary; the controller synchronizes and registers it. Do not repeat
dependency research or install packages into a shared environment.
You may inspect imports, configuration and executable versions without training.
`mcp__tsad__check_method` is an optional import diagnostic, not a completion gate.
A known missing runtime dependency is part of your integration task.

When integration is ready, finish your turn. The main process discovers compute
resources, runs the complete configured evaluation and exports the leaderboard.
Do not run training, smoke experiments or random-array tests yourself. Execution
failures return to you with their logs: fix the cause and finish your turn so the
main process retries automatically. Do not tune based on evaluation scores or
substitute synthetic scores to make execution succeed. Resolve routine setup and
compatibility choices yourself; do not pause for review or confirmation.
