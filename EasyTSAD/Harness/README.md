# EasyTSAD Harness — automated paper reproduction

A thin TSAD layer on a trimmed [DeepSeek Harness](../tsautoreproduce-harness/README.md)
(DSH) agent. Given a paper, it reproduces the method as an EasyTSAD method in its own
uv environment, evaluates it with the existing LiveLeaderBoard pipeline, fills the
leaderboard, and records every step, including human corrections, for later analysis.
See [DESIGN.md](DESIGN.md) for the architecture and module contracts.

```
scripts/harness search "anomaly transformer association discrepancy"
scripts/harness new AnomalyTransformerR --arxiv 2110.02642      # FETCH, then stops before the agent
scripts/harness run AnomalyTransformerR        # EVIDENCE -> review gate
scripts/harness log AnomalyTransformerR        # inspect contract / notes / timeline
scripts/harness approve AnomalyTransformerR    # or: correct -m "window is 100, see Sec 4.1" --run
                                                #     approve only clears an open review gate; --force skips a phase (logged)
scripts/harness run AnomalyTransformerR        # IMPLEMENT -> SMOKE -> launches EVALUATE (detached)
scripts/harness status AnomalyTransformerR     # poll the evaluation
scripts/harness run AnomalyTransformerR        # after eval: REPORT (catalog + export + report.md)
scripts/harness guard AnomalyTransformerR      # protected-file violations; --restore, or --accept -m NOTE
scripts/harness reset AnomalyTransformerR IMPLEMENT --kill-eval   # rewind; --kill-eval stops a running eval
scripts/harness publish                         # benchmark.json -> leaderboard/public, npm build
scripts/harness publish --deploy                # ... and run leaderboard/scripts/deploy_webhost.sh
```

## Setup

```
scripts/tsad sync harness        # envs/harness: dsh runtime (bundled Node), mcp, httpx, pymupdf
export yyt_api=sk-...            # name configured in config.toml [llm].api_key_env
scripts/harness doctor           # binary, key, endpoint, profile composition
```

All settings live in [config.toml](config.toml):
- `[llm]`: the model route. It is rendered into DSH's `llm-pi-ai` provider. The default is
  YYT `gpt-6.1-sol` over the Responses API. `wire_api = "chat"` switches to Chat Completions.
- `[network]`: hosts reached without the proxy.
- `[budget]`: steps, wall clock, repair rounds and smoke runs.
- `[human]`: review gates.
- `[smoke]` / `[eval]`: the runs the method has to pass.
- `[protect]`: repository paths the agent must not modify.

## Modules

| Module | Files | Role |
|---|---|---|
| Paper search & download | `tsad_harness/papers.py` | arXiv API, GitHub, LLM web search. PDF → `paper.md`, official code clone. |
| Reproduction & evaluation | `pipeline.py`, `agent.py`, `tools.py`, `validate.py`, `guard.py`, `runner.py`, `prompts/` | Phase state machine and DSH runs. The MCP tools (`mcp__tsad__*`) cover contract, notes, import check, smoke and questions. Exit validators, the protected-file guard, and the uv env + `scripts/tsad exp` integration. |
| Results & leaderboard | `report.py` | `LiveLeaderBoard/catalog.toml` entry, `scripts/tsad export`, `report.md` / `REPRODUCTION.md`, and site refresh. |
| Human correction & records | `review.py` | `approve`, `correct`, `answer` and `log`. Context snapshots go to `corrections.jsonl`. |

DSH trimming is done by profile composition, not a fork. [`dsh/trim.patch.yml`](dsh/trim.patch.yml)
removes DeepSeek upload and telemetry, web tools, subagents, goal/plan/todo, workflow/PTC, skills
and the remote gateway from the shipped `headless` profile. Per task, `agent.py` renders
`dsh/runtime.patch.yml`, which sets the LLM route, the system prompt, the session root inside the
task, and the TSAD MCP server.

## Task records (`tasks/<Name>/`)

Each task directory holds these records; DESIGN.md §6 has the details.
- `task.toml`, `state.json`: the task definition and its state.
- `journal.jsonl`: the harness log.
- `dsh/sessions/**/session.v4.jsonl`: the full agent trace.
- `dsh/runs/NNN-*.{prompt.md,ndjson}`: the prompt and events of each run.
- `contract.json`, `notes.jsonl`, `corrections.jsonl`, `questions.jsonl`: reproduction contract, notes, human corrections and agent questions.
- `runs/`: the smoke and eval specs and their outcomes.
- `report.md`: the reproduction report.

The downloaded PDF and the official code clone are git-ignored.
