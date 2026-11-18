---
name: research-environment
description: Build a method-specific locked uv environment from the official repository dependencies for the controller to install and lock with uv.
---

Read the assigned `upstream/` dependency files and imports. Resolve the Python version, framework and native package requirements actually needed by this method; record any compatibility adjustment and its source.

Write only `environment/pyproject.toml` and `reproduction/environment.json` with dependency evidence and any compatibility adjustments. The controller's environment installer then runs scripted uv lock/sync to produce `environment/uv.lock` and `.venv`, checks imports and records command/exit/log evidence. Return the proposed environment and provenance; the research worker must not invoke uv or claim installation completion before the controller checks it.

Keep installation scoped to this environment and the controller's permitted process/GPU policy. Do not modify shared environments, run training, or overwrite the hyperparameters/interface artifacts. Return the environment path, dependency evidence and any concrete incompatibility; the controller supplies lockfile and installation evidence.
