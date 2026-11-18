结论：CANDI 已确认的四轮恢复主要暴露网络、环境残缺和 Harness 故障路由问题；后两轮在 CANDI 执行前就因缺失 `matplotlib` 失败，不能据此认定算法实现有误。故障分流、环境强制重装和输出预算记忆已实现；组合验证与真实任务恢复结果见文末。

**统一记录与范围**

- **入口**：`docs/recovery/` 是恢复问题的统一记录目录；每项发现记录事实、推断、拟采取动作、原始证据与审查状态。原始日志留在任务和方法目录，通过链接引用，不复制长日志或原始分数。
- **状态**：本文记录截至 2026-11-18 的已确认发现、实际修复和验证边界；不把代码检查通过等同于完整实验完成。
- **边界**：保留官方算法参数、训练与评测预算；只调整执行配置和故障恢复。此目录供维护者查看证据，不是 worker 的额外工作流程；不新增 smoke、源码哈希验收或研究阶段。

**CANDI：四轮已确认事实**

时间均为 UTC；下表时间对应诊断交接，不等同于首次报错时间。

| 轮次 | 诊断时间 | 已确认故障与影响 | 原始证据 |
| --- | --- | --- | --- |
| 1 | 2026-11-17 07:42 | GitHub 克隆收到 HTTP 407，源码获取被代理拒绝，尚未执行 CANDI。 | [journal.jsonl:10](../../tasks/CANDI/journal.jsonl#L10) |
| 2 | 2026-11-17 10:17 | UV 安装 `yacs` 时发生 `Disk quota exceeded (os error 122)`，留下部分安装的虚拟环境。 | [journal.jsonl:43](../../tasks/CANDI/journal.jsonl#L43)、[environment-install.log:33](../../../Methods/CANDI/reproduction/environment-install.log#L33)、[安装日志后续:35](../../../Methods/CANDI/reproduction/environment-install.log#L35) |
| 3 | 2026-11-18 02:12 | `EasyTSAD.Controller → Plots → matplotlib` 导入失败，发生在 CANDI 执行前。 | [journal.jsonl:69](../../tasks/CANDI/journal.jsonl#L69) |
| 4 | 2026-11-18 02:33 | 同一 `matplotlib` 缺失再次出现，环境残缺仍未解除。 | [journal.jsonl:78](../../tasks/CANDI/journal.jsonl#L78) |

- **依赖声明**：`matplotlib`、`yacs` 已在 [环境项目](../../../Methods/CANDI/environment/pyproject.toml) 中声明；不能把这次故障简单归为漏写依赖。
- **安装残缺**：诊断记录了 dist-info 元数据存在而包文件缺失：`matplotlib` 仅有 dist-info，没有包目录；`yacs` 仅剩部分文件，缺少 `__init__.py`。证据见 [第三轮](../../tasks/CANDI/journal.jsonl#L69) 与 [第四轮](../../tasks/CANDI/journal.jsonl#L78)。元数据存在不代表包可用。
- **后续同步**：后续 `uv sync` 返回 0，输出 `Checked 87 packages`，未重新安装缺失包；检查输出见 [安装日志:76](../../../Methods/CANDI/reproduction/environment-install.log#L76) 与 [安装日志:78](../../../Methods/CANDI/reproduction/environment-install.log#L78)。退出成功与环境残缺同时存在，是本案需要处理的事实。
- **输出预算**：worker 曾以 16K 输出预算成功完成，后续阶段又回到 8K 并截断；[journal.jsonl:80](../../tasks/CANDI/journal.jsonl#L80) 记录 `max-tokens`，预算恢复事件见 [journal.jsonl:72](../../tasks/CANDI/journal.jsonl#L72) 与 [journal.jsonl:81](../../tasks/CANDI/journal.jsonl#L81)。这些是 worker 输出问题，不能计作算法失败。

**推断与待确认项**

- **UV 元数据可信度**：一种解释是后续同步依据现存安装元数据判断依赖已满足，没有发现文件残缺。这是基于上述现象的推断；尚未核查 UV 内部判断路径，不能断言其具体缓存、校验或重装机制。
- **存储因果边界**：配额错误和残缺文件均有证据，支持优先处理环境部分安装；具体配额归属、所有缺失文件的形成过程及完整影响范围仍需核查，不把历史诊断中的空间估算当成已验证容量要求。

**已确认的 Harness 问题与优化方向**

- **导入故障分流**：原流程把导入失败硬编码为实现错误；现由 [pipeline.failure_category](../../tsad_harness/pipeline.py) 统一分类，[Scheduler.finish_implementation](../../tsad_harness/scheduler.py) 将已知缺包直接交给环境恢复，不再要求实现 worker 反复修改模型。
- **诊断后重新路由**：DIAGNOSE 在调用诊断 worker 前后均检查环境、资源和发布故障，再决定是否消耗算法修复轮次；启动时只根据已有诊断证据归还历史误计轮次，保留迁移审计记录。
- **空方法名修复**：[runner.classify_failure](../../tsad_harness/runner.py) 不再使用空前缀匹配，也不把名称相近的第三方依赖视为方法模块。
- **预算持久化**：[Scheduler._run_worker](../../tsad_harness/scheduler.py) 按角色和模型保存输出预算；跨阶段、修复轮次及服务重启复用，仍遵守配置上限和现有会话恢复规则。
- **残缺环境修复**：[environments.plan_recovery/install](../../tsad_harness/environments.py) 根据依赖声明和安装元数据选择强制重装包；再次失败升级到全部依赖重装，仍失败才调用环境 worker。每轮执行原先失败的导入，环境故障最多恢复三轮；原始训练成功前不提前清零恢复计数。

**故障、动作与证据速查**

以下是恢复动作映射；没有具体事故日志的类别仅列代码入口，不表示本次已复现或已验证修复。

| 故障类别 | 恢复动作方向 | 证据或代码入口 |
| --- | --- | --- |
| 模型传输 / worker 启动 | 记录具体传输或启动错误，按现有策略恢复会话、退避重试；不消耗算法修复轮次。 | [agent.py](../../tsad_harness/agent.py)、[scheduler.py](../../tsad_harness/scheduler.py) |
| 模型输出预算 | 识别截断，持久保存预算反馈，在现有上限内继续；无进展时按既有策略接续会话。 | [截断记录:80](../../tasks/CANDI/journal.jsonl#L80)、[预算逻辑](../../tsad_harness/scheduler.py) |
| 环境部分安装 | 原导入失败 → 对应分发包强制重装 → 全量重装 → 环境 worker；每步重新检查原导入，不以 UV 成功替代恢复判定。 | [环境处理](../../tsad_harness/environments.py)、[恢复调度](../../tsad_harness/scheduler.py) |
| 环境未知依赖 | 结合异常与依赖声明交给环境角色定位；信息不足时保留未知原因，不直接归为算法错误。 | [runner.py](../../tsad_harness/runner.py)、[环境处理](../../tsad_harness/environments.py) |
| 环境存储 / 配额 | 核查可用空间、配额与缓存位置，解除阻塞后恢复安装；不能仅凭返回 0 判定残缺已修复。 | [安装失败:33](../../../Methods/CANDI/reproduction/environment-install.log#L33)、[后续检查:76](../../../Methods/CANDI/reproduction/environment-install.log#L76) |
| GPU OOM | 依据显存证据调整设备选择或等待资源，保留官方配置及训练预算。 | [runner.py](../../tsad_harness/runner.py)、[scheduler.py](../../tsad_harness/scheduler.py) |
| 发布失败 | 从同一实验已有分数恢复发布，保留成功数据集结果，避免重新训练或修改得分。 | [publicationRecovery 路径](../../tsad_harness/scheduler.py) |
| 算法实现 / 适配错误 | 仅在方法代码、接口、维度、对齐或预处理异常有证据时交实现角色修复，并使用原有算法修复预算。 | [分类器](../../tsad_harness/runner.py)、[诊断及修复计数](../../tsad_harness/scheduler.py) |

**验证与运行状态**

- **安装动作**：21 项环境恢复检查已通过，覆盖包映射、升级顺序、取消清理、错误参数和残缺环境恢复；使用隔离测试环境，没有缩短真实方法训练。
- **组合验证**：调度与环境恢复组合检查运行中；真实 CANDI 恢复尚未据此宣告完成。
