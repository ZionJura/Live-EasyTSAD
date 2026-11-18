# Harness 实现分工

`service.py` 提供鉴权后的版本化 HTTP API 和持久 SSE；TUI 和兼容 CLI 共用它；不提供浏览器页面、静态文档路由或前端 CORS，也不保留网页登录或 Cookie 会话；客户端直接携带 Bearer 令牌。`store.py` 以 SQLite 记录 Batch、Job、Attempt、AgentBinding、指标投影、诊断和资源租约，DSH 保有唯一会话日志。请求 ID 避免重复创建批次，单写者服务锁避免两个调度器启动同一工作。

`scheduler.py` 负责固定 FETCH → RESEARCH → IMPLEMENT → EVALUATE → REPORT → DONE 状态机，以及按需 DIAGNOSE、预算、并行研究、队列、取消和进程恢复。`workers.py` 定义角色和结构化产物契约，`agent.py` 调用固定版本的独立 DSH CLI，`worker_tools.py` 仅向各角色开放所分配文件的写入权限。Skills 只有获取仓库、参数研究、接口研究和环境研究四个，不负责调度。

每轮 CLI 的受限 MCP 服务由控制器提前启动，等待就绪后以 loopback Streamable HTTP 连接，使用随机端口和每轮秘密路径。冷启动 Python 导入不计入原版客户端的 60 秒 MCP 握手窗口。MCP 服务与 CLI 分别登记 PID、进程身份和日志，随本轮停止；持久记录保留恢复所需的实际端点，API、事件和 `/ps` 展示会移除秘密路径。

`pipeline.py` 获取原始仓库至方法目录 upstream/；三个研究角色交付 config.toml、hyperparameters.json、adapter.yaml 和 environment/pyproject.toml。`environments.py` 注册方法本地 UV 项目并安装。IMPLEMENT 写 BaseMethod 入口和 align.py；`runner.py` 启动统一 scripts/tsad exp，仅在实际执行错误或明确输出异常时启动诊断角色。正常训练、评测和报告不启动模型。

`dsh-host/` 固定发布包 0.2.2-rc.1，使用 base-only profile，只补入原生 Session Controller、Preset Registry、Workspace、File Reference、File Upload 和 Connection 后端服务。Connection 采用上游 SDK 的无 Webserver 组合，不挂载网页路由；不加载 `dsh-web-app` 和浏览器 UI。通过 Cordis 原生服务提供持久聊天、命令、Goal、Todo 和原生子会话。npm rc1 未发布原生 Team 包，当前 Team 使用 TSAD 项目内的 `team-adapter.mjs`：Lead 调用 `tsad_team`，状态原子保存在 `DSH_HOME/tsad-team-boards.json`，健康接口标为 `teamImplementation: tsad-adapter`。任务板仅辅助对话，不接管实验调度。流程 CLI 会话通过绑定树展示，完成后的直接聊天恢复原角色会话但只读。Host 与流程 CLI 共享模型许可；实验服务拥有方法写锁、环境锁、GPU 租约和进程停止权限。

训练 schema 通过可选 observer 发出生命周期事件并原子写曲线分数。可信增量评测复用 Performance 和既有协议，保护训练 RNG。原始数据只写 Results/Scores、Results/Evals、Results/RunTime；方法 reproduction/runs 保存执行元信息和事件。正式 all/avg 提交和榜单导出使用全局锁，最新部分运行不会遮住完整结果。导出器将同一规范化 JSON 写入两个 benchmark 目标，失败保留待恢复记录，提交后才广播 benchmark.updated。

`report.py` 的 finalize 是纯脚本目录登记和双路径导出，results_view 读取原始结果。历史 Markdown 报告函数保留为兼容工具，不在新主线调用。终端客户端使用早期原版 DSH 的 Node/pi-tui 显示组件，通过现有 API 展示阶段、分支、进度和进程；/ps 的文件内容仅来自服务拥有的工作区脚本，凭据被移除。
