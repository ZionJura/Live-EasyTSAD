# EasyTSAD Leaderboard

独立于 Python 训练环境的 React + TypeScript + Vite 前端。当前提供可交互设计原型，默认载入明确标注的合成演示数据，也支持本地导入真实结果。视觉和信息架构见 [DESIGN.md](./DESIGN.md)。

## 运行

需要 Node.js 22.12+（或 20.19+）。在仓库根目录执行：

```bash
cd leaderboard
npm ci
npm run dev
```

访问终端显示的地址（默认 http://localhost:5173）。远程开发可通过编辑器转发 5173 端口。

```bash
npm run build    # TypeScript 检查 + 构建 dist/
npm run preview  # 预览生产构建，默认 4173
```

`vite.config.ts` 使用相对 base，可将 `dist/` 部署到静态站点子目录。不依赖后端或 Python GPU 环境。Google Fonts 不可达时自动使用系统字体，不影响功能。

## webhost 部署

IP 访问地址：**http://49.232.30.3/liveadboard/**。HTTPS 地址：**https://zr-zhang25.cn/liveadboard/**。不带末尾 `/` 的地址自动跳转，并保留查询参数。

HTTP 与 HTTPS 入口共用路径转发规则，避免浏览器升级协议或缓存旧重定向后出现 404。现有 TLS 证书签发给 `zr-zhang25.cn`，HTTPS 应使用上述域名；IP 的 HTTPS 路由也能返回页面，但证书不匹配 IP。

使用现有 SSH 别名 `webhost`，在仓库根目录执行：

```bash
bash leaderboard/scripts/deploy_webhost.sh
```

脚本会构建生产文件、上传新版本、校验并平滑重载 Nginx，检查 HTTP 和 HTTPS 服务内容与本地构建一致。若当前 `dist/` 已完成构建，可传入 `--skip-build`。服务器自检失败时自动恢复原有版本、共享路径配置和域名配置；历史版本保留在服务器，便于回滚。

| 项目 | 配置 |
| --- | --- |
| SSH 主机 | `webhost`（`49.232.30.3`） |
| 公网入口 | `http://49.232.30.3/liveadboard/`，80 端口 |
| HTTPS 入口 | `https://zr-zhang25.cn/liveadboard/`，443 端口 |
| 独立站点端口 | `127.0.0.1:8766`，由 Nginx 反向代理 |
| 当前版本 | `/var/www/liveadboard/current` 软链接 |
| 历史版本 | `/var/www/liveadboard/releases/<UTC 时间戳>/` |
| 服务器配置 | `/etc/nginx/conf.d/zzz-liveadboard.conf` |
| 共享路径配置 | `/etc/nginx/snippets/liveadboard-route.conf`，由 IP HTTP 和域名 HTTPS 虚拟主机引用 |
| 配置源文件 | `deploy/webhost.nginx.conf`、`deploy/liveadboard-route.nginx.conf` |
| 访问日志 | `/var/log/nginx/liveadboard.access.log` |
| 错误日志 | `/var/log/nginx/liveadboard.error.log` |

站点由已启用开机启动的 Nginx 托管，不依赖 SSH 会话或 Vite 开发进程。HTML 每次重新验证，带哈希的静态资源长期缓存。现有域名的 HTTPS 虚拟主机只增加榜单路径的 include，保留原有 `/lab/` 等路由；该部署仍使用页面明确标注的演示数据。

## 已实现

- 单变量 / 多变量任务与对应训练范式；三种协议独立切换。
- 多选数据集、关键词搜索、方法类型筛选、分数排序、可分享的 URL 筛选状态。
- 排行榜、结果矩阵、各数据集最佳方法与完整指标详情。
- 最多四个方法的对比图；固定 0–1 坐标，保留缺失值。
- 导出当前筛选下的 CSV，保留来源、范式、指标、覆盖率和各数据集分数。
- 本地 JSON 导入、格式校验、空状态、移动端布局、键盘操作及弹窗焦点管理。

## 真实结果接入

### 当前线上分数的来源

页面默认状态来自 `src/data.ts` 的 `demoBenchmark`，**当前线上默认分数不是实验结果，也不是从参考网站或论文采集的成绩**。为设计交互，代码给 10 个方法预设基准值（如 0.917、0.892），再根据「方法 / 数据集 / 训练范式 / 指标」的字符串计算固定扰动（-0.075 至 +0.075），截断到 0–0.992 并保留四位小数。个别组合刻意设为 null，用来展示缺失数据和覆盖率。该生成方式可重复，但不具有科研意义，示例排名受到人工基准值影响。

在这些示例值之上，前端确实运行了排名逻辑：对完整覆盖所选数据集的方法求等权均值，再排序。界面快照日期也只是演示标记，不代表实验时间。顶栏「演示数据」可以点击查看来源说明。

### LiveLeaderBoard 实验流水线

仓库现有 `LiveLeaderBoard/pipeline/run.py` 会读取 `LiveLeaderBoard/experiments/*.toml`，加载实际数据并执行方法训练与测试，随后用测试标签评估保存的异常分数。每次运行在 `LiveLeaderBoard/results/runs/<method>/<schema>/<dataset>/<run_id>/` 下记录：

- `scores/*.npy`：实际模型输出的异常分数。
- `metrics.json`：逐序列指标、数据集汇总、跳过的全正常序列等。
- `manifest.json`：运行状态、随机种子、预处理、超参数、评测 margin、环境与代码版本。

`LiveLeaderBoard/pipeline/export.py` 从最新成功运行生成 `LiveLeaderBoard/results/benchmark.json`；只运行了指定部分序列的实验不进入该网页快照。这个 JSON 已采用前端支持的格式，可通过「导入结果」查看。**当前网站尚未自动加载或同步这个文件**；导入仅影响当前浏览器会话。部署脚本只发布前端静态构建，不会启动训练或发布实验快照。

### 兼容原有 avg.json

仓库已有链路：`TSADController.do_evals()` → `Results/Evals/<method>/<schema>/<dataset>/avg.json`。`Summary.to_csv()` 和根目录 `gen_site_table.py` 是现有汇总逻辑。本原型直接读取标准化 JSON，避免把 pandas 或训练依赖引入前端。

从仓库根目录执行（仅 Python 标准库）：

```bash
python3 leaderboard/scripts/export_results.py \
  --results Results/Evals \
  --output benchmark.json
```

页面底部点击「导入结果」选择文件。文件只在浏览器读取，不上传、不持久化；刷新后恢复演示数据。分享链接仅保存筛选条件，接收者需要自行导入相同快照。

当前映射严格对应以下协议：

| 前端字段 | avg.json 键 | 取值 |
| --- | --- | --- |
| `event_f1` | `event-based f1 under pa with mode log` | `f1` |
| `point_f1` | `best f1 under pa` | `f1` |
| `auprc` | `point-based auprc` | 标量 |

缺失 / NaN / Infinity 转为 `null`，非法数值拒绝导出；全缺失文件跳过。未知训练范式与重复结果报错。旧 `one_by_one` 规范为 `naive`。**EventF1PA 的 base 未写入 avg.json；本页面采用 base=3 的协议定义，导出者必须确认实验使用相同参数。** 评测 margin、预处理、随机种子等也不在 avg.json 中，正式发布前需要补充实验清单。导出时间是快照生成日期，不代表实验执行时间。

最小数据格式（数字仅作格式示例）：

```json
{
  "version": 1,
  "demo": false,
  "updatedAt": "2026-11-04",
  "methods": [{
    "id": "TranAD", "name": "TranAD", "family": "深度学习",
    "description": "本地实验方法", "color": "#738f78"
  }],
  "datasets": [{ "id": "SMD", "task": "mts", "domain": "服务器监控" }],
  "results": [{
    "method": "TranAD", "dataset": "SMD", "schema": "mts",
    "metrics": { "event_f1": 0.8, "point_f1": 0.9, "auprc": null }
  }]
}
```

方法分类是展示元数据：导出脚本为已知传统方法和 OFA 赋予类别，其余默认「深度学习」，自定义方法需人工调整。`id` 应对应仓库方法目录以正确链接源码。导入的 `updatedAt` 使用 `YYYY-MM-DD`；方法和数据集 ID 唯一，记录按 method/schema/dataset 唯一；指标为 0–1 的有限数或 null。

## 排名约定

1. 任务和训练范式隔离，不混排。
2. 使用同一个指标，对所选数据集的序列均值做等权宏平均。
3. 完整覆盖后才计算综合分数并排名。缺失结果可在矩阵和详情查看，但不以 0 填充、不参与综合排名。
4. 使用完整精度排序；真正同分采用竞赛排名（1、1、3）。展示四位小数可能造成视觉同分。
5. 搜索 / 方法类别只筛选显示内容，不改变排名；数据集 / 指标变化重算排名。
6. 数据集最佳数量允许并列，且在全部方法中计算；缺失综合排名的方法仍可能取得单数据集最佳。

## 验证

```bash
npx playwright install chromium
npm run check
python3 -m unittest discover -s checks -p 'check_*.py'
```

Playwright 检查筛选、排序、缺失值、对比、CSV、JSON 导入、移动端布局与键盘交互。Python 检查使用真实 avg.json 结构的临时文件验证导出映射。

## 目录

```text
src/App.tsx              页面、交互和弹窗
src/styles.css          视觉样式和响应式布局
src/data.ts             类型、演示数据、排名、导入校验与 CSV
scripts/export_results.py  从 Python 评测结果导出快照
checks/                 数据正确性与端到端验证
DESIGN.md               中文展示方案及后续接入设计
```
