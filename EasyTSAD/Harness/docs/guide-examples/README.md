# 离线教程配套示例

面向仓库内 DeepSeek Harness `0.2.2-rc.1` 源码；SDK、runtime、插件必须版本匹配。

- `sdk_chat.py`：持久会话、恢复与通知。运行会调用模型，需环境中的 `DEEPSEEK_API_KEY`。
- `research.patch.yml`：完整 `sdk` / `web` profile 的研究能力配置。并发四个子 Agent，深度二，Goal 最多 32 轮；这些是示例值。
- `command-plugin/` 和 `command.patch.yml`：原生 Web 自定义 `/hello-tsad`，读取会话 ID。
- `SKILL.md`：流程审查说明，可复制至项目 `.agents/skills/tsad-reproduction-review/SKILL.md`。

在仓库根目录阅读 `EasyTSAD/Harness/docs/deepseek-harness-guide.html`。示例检查包括 Python AST、YAML、JSON 和 JavaScript 语法；不代表已发起模型调用或安装插件。命令插件应安装到独立 Web profile；Python SDK 仅分派其明确支持的四组内置命令，自定义 Slash 不会自动获得 SDK 分派能力。
