# Webhost 项目导航

统一入口：<http://49.232.30.3/projects/> · <https://zr-zhang25.cn/projects/>

静态金白蓝导航页，每分钟发现服务器上的网页项目，显示简介、入口、状态、端口和修改时间。默认按修改时间倒序，未知时间排在最后；支持搜索、名称排序，以及已挂载 / 全部 / 内部服务筛选。

## 方案选择

| 方案 | 能做什么 | 本次选择 |
| --- | --- | --- |
| Nginx autoindex | 列出一个目录中的文件与时间 | 无法汇总多个服务端口、项目简介和反向代理路径 |
| 手工维护导航配置 | 可准确填写项目名、简介、链接 | 用作自动识别结果的补充，避免每次新增项目都改页面 |
| 本机服务发现 + 静态清单 | 识别正在监听的网页服务、读取现有 Nginx 路由、定期更新 | 采用；Python 标准库实现，无数据库、无常驻应用服务 |

官方依据：[Nginx autoindex](https://nginx.org/en/docs/http/ngx_http_autoindex_module.html)、[Nginx 反向代理](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy)、[systemd.timer](https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html)。

## 自动发现方式

1. `ss -H -ltnp` 读取本机正在监听的 TCP 端口；忽略 SSH、DNS、代理等已配置的基础服务。
2. `nginx -T` 读取生效配置及 include 文件。识别普通 server / location、root / alias，以及直接指向本机地址的 proxy_pass，优先选择现有的 HTTPS 域名入口。
3. 对本机发起有超时、大小限制的 HTTP GET，不使用环境代理、不跟随重定向。HTTPS 保留域名 SNI 并正常校验证书。只有成功的 2xx HTML 响应才计入“确认返回 HTML 的端口”；404 错误页、JSON API 不计入。
4. 静态项目按根目录合并，代理项目关联后端端口。同一份内容的 HTTP、HTTPS、内部预览端口不会重复生成项目卡片。项目数和 HTML 端口数分别展示。
5. 无 Nginx 路由的 HTML 服务：如果仅监听本机，标为内部服务且不生成公开链接；如果已监听外部地址，生成直接端口链接。此操作不修改防火墙、不增加反向代理；直接端口的外网可达性取决于安全组和防火墙。
6. 页面 title 和 description 用作默认介绍；`config.json` 可以提供更准确的文案。需要登录的项目使用配置简介或通用说明，不从受保护页面提取内容；链接继续经过原登录验证。
7. 原子替换 `projects.json`；浏览器每分钟刷新清单。检测失败保留上次成功结果，超过三分钟会提示清单可能已过期。页面“刷新清单”读取最新快照，不直接启动服务器扫描。

扫描器只访问本机已有服务，不扫描其他机器。`nginx -T` 需要读取现有配置，systemd 服务以 root 执行，并限制文件写入范围；网站仅提供静态文件和元数据，不提供执行扫描的公开 API。

### 修改时间

优先使用当前发布目录中网页和静态资源文件的最大 `mtime`，包含 HTML、CSS、JS、JSON、图片、字体、PDF、Markdown。跟随发布目录的 `current` 链接，但不递归进入其他符号链接；忽略隐藏目录、日志、node_modules、历史 releases，最多检查 20,000 个资源文件。

无法读取项目文件时采用 HTTP `Last-Modified`。再无可靠时间时保留未知；后续观察到页面内容指纹变化才记录“检测到内容变化”，与文件修改时间分开标注。首次发现时间和定时检测时间不会冒充修改时间。所有展示时间为北京时间 UTC+8。

这里的文件时间代表服务器当前部署文件的修改时间，是否等于源码提交时间取决于项目部署方式；动态页面的内容变化也不必然意味着代码更新。

### 识别边界

- 已实现常见 Nginx 静态站点、alias 路径和直接指向 `127.0.0.1` / `localhost` 的代理。未实现完整 Nginx 语义：命名 upstream、变量 proxy_pass、正则或嵌套 location，以及复杂 rewrite 可能需要补充适配。
- 普通独立服务探测根路径 `/`，仅在其他路径提供网页且未通过已识别的 Nginx 路由挂载的服务，可能无法自动发现。
- 401 / 403 和登录重定向不算“成功 HTML 端口”；已有静态 HTML 根目录且带认证的项目仍会作为需登录项目列出。端口统计是确认结果，不是对所有端口用途的猜测。
- 没有有效域名证书的独立 HTTPS 服务可能无法确认；不会关闭证书验证。
- 项目上线 / 下线后在下次检测更新；纯端口服务停止监听后从清单移除，保留配置且仍有 HTML 文件的 Nginx 项目可以显示暂时不可用。

## 运维与扩展

部署不需要 Node.js，只依赖已有的 Python 3.10+、curl、ss、Nginx 和 systemd：

```bash
bash web-directory/deploy.sh
```

脚本上传独立 release，原子切换软链接，备份并补充 IP HTTP 和域名 HTTPS 路由；语法或健康检查失败自动回滚。首次安装创建配置，后续部署保留服务器上的自定义配置。原 `/lab/`、`/reports/`、`/liveadboard/` 路由继续保留。

| 文件 / 服务 | 用途 |
| --- | --- |
| `/etc/project-directory/config.json` | 项目名称、简介和排除规则 |
| `/opt/project-directory/current/` | 当前代码版本 |
| `/var/www/project-directory/current/static/` | 导航页静态文件 |
| `/var/www/project-directory/data/projects.json` | 公开项目清单，不含服务器文件路径或进程参数 |
| `/var/lib/project-directory/state.json` | 本机内容指纹与首次发现记录，不对外提供 |
| `project-directory.timer` | 每次任务完成后约 60 秒再次检测 |
| `project-directory.service` | 一次性发现任务 |

常用命令：

```bash
ssh webhost systemctl status project-directory.timer
ssh webhost systemctl start project-directory.service
ssh webhost journalctl -u project-directory.service -n 20 --no-pager
```

新增项目正常部署后会被发现。需要准确简介时，在服务器配置的 `projects` 列表加入：

```json
{
  "root": "/var/www/new-project/current",
  "title": "新项目",
  "description": "一句话介绍项目的用途。",
  "category": "研究工具",
  "icon": "window"
}
```

独立端口服务用 `"port": 8080` 代替 `root`。支持 `"hidden": true` 隐藏卡片；可选图标为 `chart`、`book`、`grid`、`window`。修改配置后启动一次 service 或等待下一轮检测。隐藏卡片不改变端口观测统计。

已有榜单部署模板加入可选的目录路由 include，后续重新发布榜单不会覆盖 `/projects/` 入口。

## 验证

```bash
python3 web-directory/checks/check_discovery.py
DIRECTORY_URL=http://49.232.30.3/projects/ node web-directory/checks/check_browser.mjs
```

浏览器检查复用 `leaderboard/node_modules` 中的 Playwright，覆盖排序、搜索、筛选、手机布局、清单读取失败、恶意标题及 URL；截图写入忽略的 `artifacts/`。发现逻辑检查包含真实本地 HTTP 响应、认证项目去重、时间来源、端口解析和非 HTML API 排除。
