# TSAD Harness

恢复问题与优化记录见 [统一恢复记录](docs/recovery/README.md)。

在简洁的终端命令框中持续对话、批量复现论文。默认模型为 `deepseek-v4.1-flash`。

```bash
scripts/tsad sync harness
(cd EasyTSAD/Harness/dsh-host && npm ci)
scripts/harness prepare-tui
scripts/harness tui
```

TUI 使用恢复的原版 pi-tui 终端组件，通过现有控制 API 连接后台。`prepare-tui` 按源码与 npm 锁文件构建，将独立 Node 24 和单文件 bundle 放到本机 `/tmp/tsad-harness-tui-$UID`（目录权限 0700）。`tui` 直接运行该缓存；缺失或源码变更时自动准备，命中缓存时不会运行 npm 或启动 Python 预检查。临时缓存被清理后可重新准备；`TSAD_TUI_CACHE_DIR` 可指定当前用户的私有绝对路径。

显示层取自官方提交 `7248b5ec8f8769f882f12fd521504fa48e97bcf3`，固定 `pi-tui 0.80.7` 并保留官方编辑器补丁。源码来源、许可和适配说明见 [tui/vendor/PROVENANCE.md](tui/vendor/PROVENANCE.md)。终端包仅负责显示与远程控制，不包含另一套 Agent 运行时。

首屏由 Node 直接绘制，随后连接后台；服务未运行时由完整 Python 环境启动。退出客户端后实验继续运行。其他命令、后台和 Worker 继续使用 `envs/harness/venv`，无需独立 Python TUI 环境。需要单独运行后台时使用 `scripts/harness serve`；后台只提供 TUI 控制 API，不提供网页。

直接输入文字进行持久对话；明确的复现请求通过结构化工具进入流程。只保留四个基本操作：Enter 发送、Shift+Enter 换行、Ctrl+Q 退出客户端、Ctrl+K 停止当前回复。默认启动时新建独立会话；`scripts/harness tui --resume latest` 恢复最近活跃的主会话，`--resume <session-id>` 恢复指定会话。`/new` 新建并切换，`/new --background` 异步新建且保留当前视图（可连续提交多个），`/sessions` 查看后台会话和本客户端尚未完成或失败的创建，`/agent <session-id>` 或 `/resume <session-id>` 切换。旧会话在切换或退出后继续运行。`/connect` 重连后台；其他操作直接输入 `/` 命令。后台健康后立即显示已连接；历史与命令独立加载，单会话读取失败不会把整个后台标为断连。阶段、研究分支与训练进度在状态行中持续更新。

会话先加载最近一页记录，随后接收增量事件；`/history` 或 `/history older` 逐页加载更早历史，已加载记录保留在当前会话中。`/details on` 展开工具输出，`/details off` 折叠；这些显示命令不会启动模型或实验。

```text
/reproduce 2401.12345 2402.12345
/tasks
/agents
/agent <session-id>
/ps [job-id]
/logs <job-id>
/scores <job-id>
/report <job-id>
/pause <job-id>
/resume <job-id>
/correct <job-id> 使用官方配置中的 batch_size
/retry <job-id>
/stop <job-id>
/help
```

多行标题、URL、本地 PDF 也可提交；当前配置最多同时推进五个方法。`/stop` 停止实验，停止回复只影响所选 Agent。CLI 子会话执行中的补充排队，结束后可以恢复只读对话；修改代码通过纠正、重试和方法写锁执行。原生 `/goal`、`/plan`、`/compact` 和普通聊天能力保留，未知命令会明确报错。

CLI 的 `new --after <job>` 或 `resume --after <job>` 可持久保存执行顺序，重复传入可等待多个前置任务；前置任务完成或失败后继续后续任务。接口研究识别单变量、多变量及训练协议，控制器按方法能力选择已有 EasyTSAD 数据与评测入口；非时序异常检测或无法适配的方法报告原因后跳过。任务级评测计划和逐数据集成功记录随任务保存，恢复时读取历史 manifest，成功数据集不因其他数据集失败而重跑；相关数据集配置改变时重新执行。

原生可继续子 Agent 可通过 `/agent <session-id>` 进入并续聊，后台重启后也能恢复一级子会话。上游 `0.2.2-rc.1` 没有仅恢复子 Agent 控制器的接口；重启后直接恢复二级子会话时，需要先真实续聊它的直接父 Agent。后台会指出需要恢复的父会话，不会自动发送空消息唤醒模型。

`new --evaluation-spec <experiment.toml>` 可复用原生实验配置指定该任务的数据类型、训练协议、数据集和评测指标；跨数据集协议使用显式 `src_datasets`、`dst_datasets`，没有指定配置时由接口能力选择适用路径。

## 固定流程

```text
FETCH → RESEARCH（参数、接口、环境三个 CLI 子 Agent 并行）
      → IMPLEMENT → EVALUATE → REPORT → DONE
出现执行错误或明确的输出异常 → DIAGNOSE → 按证据修复并重试
```

FETCH 子 Agent 查明论文与官方仓库，脚本负责下载与克隆；暂不支持没有源码仓库的方法。RESEARCH 的三个角色分别交付结构化产物，脚本检查后安装方法本地 UV 环境。IMPLEMENT 编写 BaseMethod 适配。正常的 EVALUATE 和 REPORT 完全由脚本执行，不调用模型。诊断只为执行、维度、对齐、预处理或接口错误安排修复；低分本身不授权改算法、指标或按测试得分调参。

只有四个 Skill：`fetch-repository`、`research-hyperparameters`、`research-interface`、`research-environment`。主线和阶段推进保留在 Harness 代码中。

每轮 CLI Worker 启动前，控制器先预启动该角色的受限 MCP 服务，等待就绪后通过 loopback Streamable HTTP 接入。服务只监听 `127.0.0.1`，使用每轮随机端口和秘密路径；冷启动导入不占用原版客户端的 60 秒 MCP 握手时间。该服务随本轮结束而停止，进程、日志和身份记录可通过 `/ps` 查看；对外展示会移除秘密路径。

Worker 会话使用独立的 `worker-sessions` 目录，与交互宿主的压缩会话隔离；恢复旧会话时复制对应会话并保留原压缩格式。启动失败记录退出码与日志，不计入算法修复轮次；模型代理的临时连接错误在原会话中退避重试，最多五次，重试状态随任务保存。实现 worker 可在自己的方法目录内完成必要的源码、适配和环境修复，共享评测与其他方法不在写入范围内。

```text
EasyTSAD/Methods/<Method>/
  upstream/                         # 原仓库结构
  paper/{paper.pdf,paper.md}
  reproduction/
    source.json                     # 来源、commit、相对路径
    hyperparameters.json            # 参数值与证据
    adapter.yaml                    # source / target / align / evidence
    diagnostics.jsonl               # 诊断和修复记录
    agents/                         # 角色会话、CLI 事件日志
    runs/<attempt>/<schema>/<dataset>/ # manifest / progress / 日志
  environment/{pyproject.toml,uv.lock,.venv/}
  config.toml
  align.py                          # 仅在适配器需要独立对齐助手时提供
  <Method>.py
  __init__.py
```

研究超参数角色生成原生 `config.toml`，接口角色描述模型与 EasyTSAD 的输入输出及转换要求，环境角色生成 UV 项目。`adapter.yaml` 是实现交接说明，实际转换按需写入方法适配器，可抽取到 `align.py`，不执行 YAML 中的任意逻辑。新环境在注册表记录方法本地项目；`scripts/tsad` 继续兼容既有 `envs/` 项目。

原始实验结果只写标准 EasyTSAD 布局：

```text
Results/Scores/<Method>/<schema>/<dataset>/*.npy
Results/Evals/<Method>/<schema>/<dataset>/{all.json,avg.json}
Results/RunTime/<Method>/<schema>/<dataset>/
```

每条曲线完成后显示暂定指标，完整数据集成功后发布正式结果。榜单导出统一更新 `LiveLeaderBoard/results/benchmark.json` 和 `leaderboard/public/benchmark.json`，两处内容一致并有失败恢复记录；历史运行仍可读取。REPORT 不强制产生额外 Markdown 或重复原始分数。独立的 React 排行榜继续展示导出的结果，刷新即可加载新的快照；Harness 不再提供网页交互。公网榜单部署需显式 `/publish --deploy`。

## 配置与验证

`config.toml` 管理模型代理、检索、数据集、GPU 和预算。密钥从环境变量或用户私有文件读取。`[search].base_url` 可配置支持 JSON 的 SearXNG，论文检索和 HTTP/PDF 下载独立工作。原有 `new`、`run`、`status`、`correct`、`publish` 等命令保留；新交互运行时固定为发布的 `@deepseek-ai/dsh@0.2.2-rc.1`，依赖锁随代码提供。旧 Python SDK 仅为历史工具兼容保留。

执行配置依据反馈调整并保存在任务日志和状态库中：输出截断后逐次增加输出预算（上限为 `llm.max_recovery_output_tokens`），连续没有文件进展时使用新会话接续已有文件；依赖安装从单路起步，两次成功后提高并发、失败后降低，上限为 `interactive.max_environment_installs`，资源紧张时保持单路。GPU 从可确认空闲的设备中选择，CUDA 显存不足时提高下一次设备的空闲显存要求，保留官方训练轮数、批大小和评测定义；资源不足时等待并明确报告。依赖错误交回环境 worker，榜单发布失败从同一实验已有分数恢复，均不消耗算法修复轮次。

常驻 Host 只加载上游 `dsh-base` 和会话控制必需的原生后端服务，不加载 `dsh-web-app`、原版 Webserver 或 UI 插件，不监听原版网页端口。原生 Agent loop、会话、Goal 和子 Agent 保持复用。关闭 HMR 和 Skills 文件监听，修改 Skill 后重启后台生效。

固定 npm rc1 未发布原生 Team 包。当前 Team 是 TSAD 项目内的 `team-adapter.mjs` 持久任务板，Lead 通过 `tsad_team` 使用，状态保存在 `DSH_HOME/tsad-team-boards.json`；它辅助对话，实验调度仍由 Python 控制器负责。

```bash
scripts/harness doctor
PYTHONPATH=EasyTSAD/Harness:. envs/harness/venv/bin/python -m unittest discover -s EasyTSAD/Harness/tests
cd EasyTSAD/Harness/dsh-host && npm test
```

后台状态在 `runtime/harness-workbench/`，DSH 自己管理会话日志，SQLite 管理任务、租约和可重放事件。恢复时核对进程身份，暂停和取消状态不会自动恢复；每个方法只允许一个代码写者。GPU 租约与模型并发独立，当前配置最多五个模型请求、四轮算法修复、每轮 120 步和 90 分钟。

[离线 DeepSeek Harness 中文教学报告](docs/deepseek-harness-guide.html) 包含上游能力索引、配置、源码证据、教程和当前 TSAD 流程。另见 [DESIGN.md](DESIGN.md)。
