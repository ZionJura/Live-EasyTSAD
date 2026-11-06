"""TSAD paper-reproduction harness built on a trimmed DeepSeek Harness (DSH).

The DSH agent loop (model -> tools -> model, sessions, compaction) is used as is
through the `dsh` binary shipped by `deepseek-harness-runtime-bin`. This package
adds the TSAD domain layer around it: paper ingestion, a deterministic phase
state machine, a file guard, EasyTSAD/uv integration, leaderboard reporting
and a human-correction journal.
"""

__version__ = "0.1.0"
