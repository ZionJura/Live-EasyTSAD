# Native DSH bridge v1

Start with `node EasyTSAD/Harness/dsh-host/launch.mjs`. Install locked packages
first with `npm ci` in this directory. The launcher uses its pinned Node 24 runtime.
The bridge is an authenticated, loopback-only HTTP service; every request,
including health, requires `Authorization: Bearer $TSAD_BRIDGE_TOKEN`.
The launcher composes `dsh-base` and native session-control services only; it
does not mount `dsh-web-app`, browser plugins, a frontend server or a Web port.

Environment: `DSH_HOME`, `TSAD_BRIDGE_PORT` (8768), `TSAD_BRIDGE_TOKEN`,
`TSAD_CONTROL_URL` (`http://127.0.0.1:8767`), `TSAD_CONTROL_TOKEN`,
`TSAD_MODEL_PROVIDER` (`terra`), `TSAD_MODEL`, `TSAD_MODEL_BASE_URL`,
`TSAD_MODEL_API_KEY_ENV` (`TERRA_PROXY_API_KEY`), `TSAD_MODEL_API`
(`openai-completions`), `TSAD_SEARCH_URL` (optional SearXNG JSON endpoint),
`TSAD_ROOT`, `TSAD_MAX_MODEL_CONCURRENCY` (4).

| Method | Path | Input / output |
|---|---|---|
| GET | `/health` | Runtime version, epoch, readiness and enabled capabilities |
| GET | `/sessions` | `{sessions:[...]}` persisted root conversations |
| POST | `/sessions` | `{sessionId?,cwd?,systemPrompt?,maxSteps?}` → `{sessionId}` |
| GET | `/sessions/:id` | `{sessionId,running,status,lastError,history,projections,goal,todos,team,binding}` |
| GET | `/sessions/:id/events?after=N&epoch=...` | `{events:[{seq,type,data}],cursor,epoch,reset}`; process-local cursor, use snapshot after reset |
| POST | `/sessions/:id/messages` | `{text,mode:'queue'|'steer',requestId?}` → 202 `{sessionId,requestId,messageId?}` |
| POST | `/sessions/:id/cancel` | Native cancellation; `{accepted:true}` |
| POST | `/sessions/:id/goals` | `{action:'set'|'disarm'|'resume'|'complete'|'pause',objective?,maxGoalRounds?}` → native goal view |
| POST | `/sessions/:id/team` | `{action:'enable'|'create'|'list'|'update',enabled?,task?,...}` → TSAD local task-board result |
| GET | `/agents?sessionId=...` | `{agents:[...]}` root plus native recursive child catalog |
| GET | `/commands?sessionId=...` | `{commands:[...]}` exact agent's native command descriptors |
| POST | `/commands` | `{sessionId,line}` → native `{commandId,result}` |
| POST | `/roles` | `{parentSessionId,role,jobId?,prompt,cwd?,maxSteps?,name?}` → `{sessionId,name,role,jobId}` |

Roles: `lead`, `researcher`, `implementer`, `supervisor`, `reporter`.
Role bindings are durable. Supervisor/researcher/reporter presets expose read
tools and domain tools only; supervisor cannot obtain shell, write, or delegation
tools. Implementer mutation requires the Python domain service's lease approval.
Native model-authored goal completion is denied for job-bound agents; only the
domain scheduler can invoke the host goal completion endpoint.

Domain tool calls use `POST $TSAD_CONTROL_URL/internal/tools/{tool}` with
Bearer control token and `{arguments,sessionId,role,jobId}`. The response is JSON.
The tool names are `reproduce,tasks,run,pause,resume,stop,retry,correct,scores,search,
fetch,status,diagnose,complete`. Inputs are structured tool arguments; native
slash handlers send `{input: rawInput}` to the same domain entry.
