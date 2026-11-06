/// <reference types="vite/client" />
import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import {
  Activity,
  ArrowDown,
  ArrowDownToLine,
  ArrowRight,
  ArrowUp,
  ArrowUpRight,
  BarChart3,
  BookOpen,
  Check,
  CheckCheck,
  ChevronDown,
  CircleHelp,
  Code2,
  Database,
  ExternalLink,
  FlaskConical,
  GitCompareArrows,
  Github,
  Info,
  Layers3,
  LayoutGrid,
  ListFilter,
  Menu,
  Search,
  Share2,
  SlidersHorizontal,
  Sparkles,
  Trophy,
  Upload,
  X,
} from "lucide-react";
import {
  demoBenchmark,
  docsUrl,
  metrics,
  rankMethods,
  repoUrl,
  resultsCsv,
  schemas,
  validateBenchmark,
} from "./data";
import type { Benchmark, Metric, Schema, Task } from "./data";

type View = "ranking" | "matrix";
interface Filters {
  task: Task;
  schema: Schema;
  metric: Metric;
  datasets: string[];
  view: View;
  search: string;
  family: string;
  ascending: boolean;
}
const taskNames = { mts: "多变量异常检测", uts: "单变量异常检测" };
const score = (value: number | null | undefined) =>
  value == null ? "—" : value.toFixed(4);

function initialFilters(): Filters {
  const p = new URLSearchParams(window.location.search);
  const task: Task = p.get("task") === "uts" ? "uts" : "mts";
  const requestedSchema = p.get("schema") as Schema;
  const schema =
    task === "mts"
      ? "mts"
      : Object.hasOwn(schemas, requestedSchema) && requestedSchema !== "mts"
        ? requestedSchema
        : "naive";
  const allowed = demoBenchmark.datasets
    .filter((d) => d.task === task)
    .map((d) => d.id);
  const selected = p
    .get("datasets")
    ?.split(",")
    .filter((d) => allowed.includes(d));
  const metric = p.get("metric") as Metric;
  return {
    task,
    schema,
    metric: Object.hasOwn(metrics, metric) ? metric : "event_f1",
    datasets: selected?.length ? selected : allowed,
    view: p.get("view") === "matrix" ? "matrix" : "ranking",
    search: p.get("q") ?? "",
    family: ["深度学习", "传统方法", "预训练模型"].includes(
      p.get("family") ?? "",
    )
      ? p.get("family")!
      : "all",
    ascending: p.get("sort") === "asc",
  };
}

function Dialog({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current!;
    const previousFocus = document.activeElement as HTMLElement | null;
    el.showModal();
    return () => {
      el.close();
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, []);
  return (
    <dialog
      ref={ref}
      className={`dialog ${wide ? "dialog-wide" : ""}`}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      aria-labelledby="dialog-title"
    >
      <div className="dialog-heading">
        <div>
          <span className="eyebrow">EASYTSAD / BENCHMARK</span>
          <h2 id="dialog-title">{title}</h2>
        </div>
        <button className="icon-button" aria-label="关闭弹窗" onClick={onClose}>
          <X size={20} />
        </button>
      </div>
      {children}
    </dialog>
  );
}

function MethodIcon({
  row,
  small = false,
}: {
  row: { color: string; name: string };
  small?: boolean;
}) {
  return (
    <span
      className={`method-icon ${small ? "small" : ""}`}
      style={{ "--method-color": row.color } as CSSProperties}
    >
      <Activity size={small ? 14 : 18} />
    </span>
  );
}

function SignalArt() {
  return (
    <div className="signal-art" aria-hidden="true">
      <svg viewBox="0 0 440 155" fill="none">
        <defs>
          <pattern
            id="grid"
            width="22"
            height="22"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="1" cy="1" r="1" fill="#d6dfec" />
          </pattern>
          <linearGradient id="fade">
            <stop stopColor="var(--blue-softest)" stopOpacity="0" />
            <stop offset=".4" stopColor="var(--blue-softest)" />
            <stop offset="1" stopColor="var(--blue-softest)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <rect width="440" height="155" fill="url(#grid)" />
        <rect
          x="238"
          y="10"
          width="58"
          height="131"
          rx="7"
          fill="var(--gold-border)"
          fillOpacity=".6"
        />
        <path
          d="M0 91 L16 91 L23 83 L31 98 L43 73 L54 93 L64 88 L75 89 L84 81 L93 97 L105 88 L119 89 L130 75 L142 99 L156 84 L170 91 L183 90 L193 78 L203 95 L213 87 L230 88 L241 67 L250 80 L260 31 L270 113 L281 48 L293 89 L307 88 L318 81 L331 97 L344 86 L358 90 L369 80 L381 95 L395 86 L407 91 L422 88 L440 91"
          stroke="var(--blue)"
          strokeWidth="2"
          strokeLinejoin="round"
        />
        <path
          d="M0 116 Q20 102 40 115 T80 111 T120 115 T160 112 T200 115 T240 109 T280 114 T320 110 T360 114 T400 112 T440 115"
          stroke="#a9bdd7"
          strokeWidth="1.5"
        />
        <circle
          cx="260"
          cy="31"
          r="5"
          fill="var(--gold)"
          stroke="var(--blue-softest)"
          strokeWidth="3"
        />
        <path d="M260 24V11h48" stroke="var(--gold)" strokeDasharray="3 3" />
        <text
          x="315"
          y="15"
          fill="var(--gold-deep)"
          fontSize="9"
          fontFamily="monospace"
        >
          DETECT THE DIFFERENCE
        </text>
      </svg>
      <span>EVERY SIGNAL TELLS A STORY.</span>
    </div>
  );
}

export default function App() {
  const [data, setData] = useState<Benchmark>(demoBenchmark);
  const [filters, setFilters] = useState<Filters>(initialFilters);
  const [selected, setSelected] = useState<string[]>([]);
  const [modal, setModal] = useState<
    "methodology" | "datasets" | "compare" | "import" | null
  >(null);
  const [detailId, setDetailId] = useState<string | null>(null);
  const [datasetOpen, setDatasetOpen] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const [toast, setToast] = useState("");
  const [importError, setImportError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const datasetRef = useRef<HTMLDivElement>(null);
  const datasetButtonRef = useRef<HTMLButtonElement>(null);

  const availableDatasets = data.datasets.filter(
    (d) => d.task === filters.task,
  );
  const rows = useMemo(
    () => rankMethods(data, filters.schema, filters.metric, filters.datasets),
    [data, filters.schema, filters.metric, filters.datasets],
  );
  const displayed = rows
    .filter(
      (row) =>
        (filters.family === "all" || row.family === filters.family) &&
        `${row.name} ${row.id}`
          .toLowerCase()
          .includes(filters.search.toLowerCase()),
    )
    .sort((a, b) => {
      if (a.score === null) return b.score === null ? 0 : 1;
      if (b.score === null) return -1;
      return filters.ascending ? a.score - b.score : b.score - a.score;
    });
  const leader = rows.find((row) => row.score !== null);
  const compared = rows.filter((row) => selected.includes(row.id));
  const detail = rows.find((row) => row.id === detailId);
  const complete = rows.filter((row) => row.score !== null).length;
  const update = (next: Partial<Filters>) =>
    setFilters((current) => ({ ...current, ...next }));

  // Load published results if the site ships a benchmark.json; keep the demo otherwise.
  useEffect(() => {
    let cancelled = false;
    fetch(`${import.meta.env.BASE_URL}benchmark.json`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then((json) => {
        if (cancelled) return;
        const next = validateBenchmark(json);
        const fromUrl = new URLSearchParams(window.location.search)
          .get("datasets")
          ?.split(",");
        setData(next);
        setFilters((current) => {
          const taskOf = (id: string) =>
            next.datasets.find((d) => d.id === id)!.task;
          const first =
            next.results.find((r) => taskOf(r.dataset) === current.task) ??
            next.results[0];
          const task = taskOf(first.dataset);
          const schema =
            task === current.task &&
            next.results.some((r) => r.schema === current.schema)
              ? current.schema
              : first.schema;
          const allowed = next.datasets
            .filter((d) => d.task === task)
            .map((d) => d.id);
          const kept = (fromUrl ?? current.datasets).filter((d) =>
            allowed.includes(d),
          );
          return {
            ...current,
            task,
            schema,
            datasets: kept.length ? kept : allowed,
          };
        });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const p = new URLSearchParams({
      task: filters.task,
      schema: filters.schema,
      metric: filters.metric,
      view: filters.view,
      datasets: filters.datasets.join(","),
    });
    if (filters.search) p.set("q", filters.search);
    if (filters.family !== "all") p.set("family", filters.family);
    if (filters.ascending) p.set("sort", "asc");
    window.history.replaceState(null, "", `${window.location.pathname}?${p}`);
  }, [filters]);
  useEffect(() => {
    if (toast) {
      const timer = window.setTimeout(() => setToast(""), 4000);
      return () => clearTimeout(timer);
    }
  }, [toast]);
  useEffect(() => {
    if (!datasetOpen) return;
    const click = (e: MouseEvent) => {
      if (!datasetRef.current?.contains(e.target as Node))
        setDatasetOpen(false);
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setDatasetOpen(false);
        datasetButtonRef.current?.focus();
      }
    };
    document.addEventListener("mousedown", click);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", click);
      document.removeEventListener("keydown", key);
    };
  }, [datasetOpen]);

  function changeTask(task: Task) {
    update({
      task,
      schema: task === "mts" ? "mts" : "naive",
      datasets: data.datasets.filter((d) => d.task === task).map((d) => d.id),
      search: "",
      family: "all",
    });
    setSelected([]);
    setNavOpen(false);
    setDatasetOpen(false);
  }
  function toggleDataset(id: string) {
    if (filters.datasets.includes(id) && filters.datasets.length === 1) {
      setToast("请至少保留一个数据集");
      return;
    }
    update({
      datasets: filters.datasets.includes(id)
        ? filters.datasets.filter((d) => d !== id)
        : [...filters.datasets, id],
    });
  }
  function toggleSelected(id: string) {
    if (selected.includes(id))
      setSelected(selected.filter((item) => item !== id));
    else if (selected.length < 4) setSelected([...selected, id]);
    else setToast("最多同时对比 4 个方法");
  }
  function resetFilters() {
    update({
      datasets: availableDatasets.map((d) => d.id),
      metric: "event_f1",
      search: "",
      family: "all",
      ascending: false,
    });
  }
  function download() {
    const blob = new Blob(
      [
        resultsCsv(
          displayed,
          filters.datasets,
          filters.metric,
          filters.schema,
          data.demo,
        ),
      ],
      { type: "text/csv;charset=utf-8;" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `easytsad-${data.demo ? "demo-" : ""}${filters.task}-${filters.schema}-${filters.metric}.csv`;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setToast("已导出当前筛选下的完整结果");
  }
  async function share() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setToast(
        data.demo
          ? "已复制链接，打开后将保留当前筛选条件"
          : "已复制筛选链接；查看者还需导入相同的结果文件",
      );
    } catch {
      setToast("无法自动复制，请复制浏览器地址栏中的链接");
    }
  }
  async function importFile(file?: File) {
    if (!file) return;
    setImportError("");
    try {
      if (file.size > 10 * 1024 * 1024)
        throw new Error("请选择小于 10 MB 的结果文件。");
      const next = validateBenchmark(JSON.parse(await file.text()));
      const first = next.results[0];
      const task = next.datasets.find((d) => d.id === first.dataset)!.task;
      setData(next);
      setSelected([]);
      update({
        task,
        schema: first.schema,
        metric: "event_f1",
        datasets: next.datasets.filter((d) => d.task === task).map((d) => d.id),
        search: "",
        family: "all",
      });
      setModal(null);
      setToast("结果已载入当前会话；刷新后恢复演示数据");
    } catch (error) {
      setImportError(
        error instanceof SyntaxError
          ? "JSON 格式无法解析，请检查文件内容。"
          : (error as Error).message,
      );
    }
    if (fileRef.current) fileRef.current.value = "";
  }

  const nav = (
    <>
      <a className="brand" href="./" aria-label="EasyTSAD 首页">
        <span className="brand-symbol">
          <Activity size={25} />
        </span>
        <span>
          Easy<span className="brand-light">TSAD</span>
          <small>TIME SERIES INTELLIGENCE</small>
        </span>
      </a>
      <div className="nav-section-label">WORKSPACE</div>
      <button
        className="nav-item active"
        onClick={() => {
          update({ view: "ranking" });
          setNavOpen(false);
        }}
      >
        <Trophy size={18} />
        Leaderboard
        <span className="nav-indicator" />
      </button>
      <button
        className={`nav-sub ${filters.task === "mts" ? "selected" : ""}`}
        onClick={() => changeTask("mts")}
      >
        <span />
        多变量异常检测<small>MTS</small>
      </button>
      <button
        className={`nav-sub ${filters.task === "uts" ? "selected" : ""}`}
        onClick={() => changeTask("uts")}
      >
        <span />
        单变量异常检测<small>UTS</small>
      </button>
      <button
        className="nav-item"
        onClick={() => {
          update({ view: "matrix" });
          setNavOpen(false);
        }}
      >
        <LayoutGrid size={18} />
        结果总览
      </button>
      <button
        className="nav-item"
        onClick={() => {
          setModal("datasets");
          setNavOpen(false);
        }}
      >
        <Database size={18} />
        数据集
      </button>
      <div className="nav-section-label resources-label">RESOURCES</div>
      <button
        className="nav-item"
        onClick={() => {
          setModal("methodology");
          setNavOpen(false);
        }}
      >
        <FlaskConical size={18} />
        评测方法
      </button>
      <a className="nav-item" href={docsUrl} target="_blank" rel="noreferrer">
        <BookOpen size={18} />
        使用文档
        <ArrowUpRight size={14} className="nav-external" />
      </a>
      <a className="nav-item" href={repoUrl} target="_blank" rel="noreferrer">
        <Github size={18} />
        GitHub
        <ArrowUpRight size={14} className="nav-external" />
      </a>
      <div className="sidebar-bottom">
        <div className="contribute-card">
          <span className="contribute-icon">
            <Code2 size={19} />
          </span>
          <strong>让好方法被看见。</strong>
          <p>
            在统一的评测流程下，
            <br />
            探索你的算法表现。
          </p>
          <a href={`${repoUrl}#usage`} target="_blank" rel="noreferrer">
            运行你的方法 <ArrowUpRight size={15} />
          </a>
        </div>
        <div className="sidebar-footer">
          <span className="status-dot" />
          Open research. Shared progress.<span>EasyTSAD · v0.3</span>
        </div>
      </div>
    </>
  );

  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        跳转到榜单
      </a>
      <aside className="sidebar">{nav}</aside>
      {navOpen && (
        <Dialog title="导航" onClose={() => setNavOpen(false)}>
          <nav className="mobile-nav">{nav}</nav>
        </Dialog>
      )}
      <div className="workspace">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              onClick={() => setNavOpen(true)}
              aria-label="打开导航"
            >
              <Menu size={20} />
            </button>
            <span>Benchmark</span>
            <span className="breadcrumb-divider">/</span>
            <strong>Leaderboard</strong>
          </div>
          <div className="topbar-actions">
            <button
              className="preview-tag"
              onClick={() => setModal("methodology")}
              aria-label="查看数据来源"
            >
              <span />
              {data.demo ? "演示数据 · Demo" : "已导入结果"}
              <CircleHelp size={12} />
            </button>
            <a
              href={repoUrl}
              target="_blank"
              rel="noreferrer"
              aria-label="查看 GitHub 仓库"
            >
              <Github size={19} />
            </a>
            <span className="topbar-divider" />
            <span className="language-tag">简体中文</span>
          </div>
        </header>
        <main id="main">
          <section className="hero">
            <div>
              <div className="eyebrow">
                <span className="tiny-line" />
                THE EASYTSAD LEADERBOARD
              </div>
              <h1>
                时序异常检测榜单<span className="title-dot">.</span>
              </h1>
              <p>看见算法的真实表现，找到适合你的时序方法。</p>
              <div className="hero-meta">
                <span>
                  <Layers3 size={14} />
                  统一评测框架
                </span>
                <span>
                  <Code2 size={14} />
                  开放、可复现
                </span>
                <button onClick={() => setModal("methodology")}>
                  了解评测方法 <ArrowUpRight size={13} />
                </button>
              </div>
            </div>
            <SignalArt />
          </section>

          <div className="summary-strip">
            <div>
              <span className="stat-icon">
                <Code2 size={19} />
              </span>
              <div>
                <strong>{rows.length.toString().padStart(2, "0")}</strong>
                <span>参评方法</span>
              </div>
            </div>
            <div>
              <span className="stat-icon">
                <Database size={19} />
              </span>
              <div>
                <strong>
                  {filters.datasets.length.toString().padStart(2, "0")}
                </strong>
                <span>所选数据集</span>
              </div>
            </div>
            <div>
              <span className="stat-icon">
                <SlidersHorizontal size={19} />
              </span>
              <div>
                <strong>
                  {Object.keys(metrics).length.toString().padStart(2, "0")}
                </strong>
                <span>评测指标</span>
              </div>
            </div>
            <div className="snapshot-stat">
              <span className="stat-icon">
                <Activity size={19} />
              </span>
              <div>
                <strong>{data.updatedAt.replaceAll("-", ".")}</strong>
                <span>
                  {data.demo ? "演示快照 · 非实测结果" : "导入快照日期"}
                </span>
              </div>
            </div>
          </div>

          <div className="task-tabs" aria-label="检测任务">
            {(["mts", "uts"] as Task[]).map((task) => (
              <button
                key={task}
                aria-pressed={filters.task === task}
                className={filters.task === task ? "active" : ""}
                onClick={() => changeTask(task)}
              >
                {task === "mts" ? (
                  <Layers3 size={17} />
                ) : (
                  <Activity size={17} />
                )}
                {taskNames[task]}
                <span>{task.toUpperCase()}</span>
              </button>
            ))}
            <div className="task-caption">TIME SERIES ANOMALY DETECTION</div>
          </div>

          <section className="filter-panel" aria-label="榜单筛选">
            <div className="filter-select">
              <label htmlFor="schema">训练范式</label>
              <div className="select-shell">
                <Layers3 size={15} />
                <select
                  id="schema"
                  value={filters.schema}
                  onChange={(e) => {
                    update({ schema: e.target.value as Schema });
                    setSelected([]);
                  }}
                >
                  {Object.entries(schemas)
                    .filter(([key]) =>
                      filters.task === "mts" ? key === "mts" : key !== "mts",
                    )
                    .map(([key, label]) => (
                      <option value={key} key={key}>
                        {label}
                      </option>
                    ))}
                </select>
                <ChevronDown size={14} />
              </div>
            </div>
            <div className="filter-select">
              <label htmlFor="metric">
                评测指标{" "}
                <button
                  className="help-button"
                  onClick={() => setModal("methodology")}
                  aria-label="查看指标说明"
                >
                  <CircleHelp size={12} />
                </button>
              </label>
              <div className="select-shell">
                <BarChart3 size={15} />
                <select
                  aria-label="评测指标"
                  id="metric"
                  value={filters.metric}
                  onChange={(e) => update({ metric: e.target.value as Metric })}
                >
                  {Object.entries(metrics).map(([key, value]) => (
                    <option value={key} key={key}>
                      {value.label}
                    </option>
                  ))}
                </select>
                <ChevronDown size={14} />
              </div>
            </div>
            <div className="filter-select dataset-filter" ref={datasetRef}>
              <span className="field-label">数据集范围</span>
              <button
                ref={datasetButtonRef}
                className="select-shell dataset-trigger"
                aria-expanded={datasetOpen}
                aria-controls="dataset-picker"
                onClick={() => setDatasetOpen(!datasetOpen)}
              >
                <Database size={15} />
                <span>
                  {filters.datasets.length === availableDatasets.length
                    ? "全部数据集"
                    : "自定义数据集"}
                  <small>{filters.datasets.length}</small>
                </span>
                <ChevronDown size={14} />
              </button>
              {datasetOpen && (
                <div className="dataset-popover" id="dataset-picker">
                  <div className="popover-title">
                    <strong>选择数据集</strong>
                    <button
                      onClick={() =>
                        update({ datasets: availableDatasets.map((d) => d.id) })
                      }
                    >
                      全选
                    </button>
                  </div>
                  {availableDatasets.map((d) => (
                    <label key={d.id}>
                      <input
                        type="checkbox"
                        checked={filters.datasets.includes(d.id)}
                        onChange={() => toggleDataset(d.id)}
                      />
                      <span>
                        {d.id}
                        <small>{d.domain}</small>
                      </span>
                    </label>
                  ))}
                  <div className="popover-foot">
                    至少选择一个 · 数据集等权汇总
                  </div>
                </div>
              )}
            </div>
            <div className="filter-right">
              <span className="direction-note">
                <ArrowUp size={13} />
                分数越高越好
              </span>
              <button className="text-button" onClick={resetFilters}>
                重置筛选
              </button>
            </div>
          </section>

          <div className="results-heading">
            <div className="view-tabs" aria-label="结果视图">
              <button
                aria-pressed={filters.view === "ranking"}
                className={filters.view === "ranking" ? "active" : ""}
                onClick={() => update({ view: "ranking" })}
              >
                <Trophy size={16} />
                排行榜
              </button>
              <button
                aria-pressed={filters.view === "matrix"}
                className={filters.view === "matrix" ? "active" : ""}
                onClick={() => update({ view: "matrix" })}
              >
                <LayoutGrid size={16} />
                结果矩阵
              </button>
            </div>
            <div className="result-actions">
              <button className="quiet-button" onClick={share}>
                <Share2 size={15} />
                <span>分享</span>
              </button>
              <button
                className="outline-button"
                onClick={download}
                disabled={!displayed.length}
              >
                <ArrowDownToLine size={15} />
                导出 CSV
              </button>
            </div>
          </div>

          <div
            className={`results-layout ${filters.view === "matrix" ? "matrix-layout" : ""}`}
          >
            <section
              className="ranking-card"
              aria-label={
                filters.view === "ranking" ? "方法排行榜" : "数据集结果矩阵"
              }
            >
              <div className="table-toolbar">
                <label className="search-box">
                  <Search size={16} />
                  <input
                    aria-label="搜索方法"
                    placeholder="搜索方法名称…"
                    value={filters.search}
                    onChange={(e) => update({ search: e.target.value })}
                  />
                  {filters.search && (
                    <button
                      aria-label="清除搜索"
                      onClick={() => update({ search: "" })}
                    >
                      <X size={13} />
                    </button>
                  )}
                </label>
                <div className="family-filter">
                  <ListFilter size={15} />
                  <select
                    aria-label="方法类型"
                    value={filters.family}
                    onChange={(e) => update({ family: e.target.value })}
                  >
                    <option value="all">全部方法</option>
                    <option>深度学习</option>
                    <option>传统方法</option>
                    <option>预训练模型</option>
                  </select>
                  <ChevronDown size={12} />
                </div>
              </div>
              <div className="mobile-table-hint">左右滑动查看完整指标 →</div>
              <div className="table-scroll">
                <table
                  className={filters.view === "matrix" ? "matrix-table" : ""}
                >
                  <thead>
                    <tr>
                      <th scope="col" className="checkbox-cell">
                        <GitCompareArrows size={14} aria-label="选择对比方法" />
                      </th>
                      <th scope="col" className="rank-cell">
                        排名
                      </th>
                      <th scope="col" className="method-cell">
                        方法{" "}
                        <span className="count-badge">{displayed.length}</span>
                      </th>
                      <th
                        scope="col"
                        className="score-cell"
                        aria-sort={
                          filters.ascending ? "ascending" : "descending"
                        }
                      >
                        <button
                          className="sort-button"
                          onClick={() =>
                            update({ ascending: !filters.ascending })
                          }
                        >
                          {filters.view === "matrix"
                            ? "均值"
                            : metrics[filters.metric].short}
                          {filters.ascending ? (
                            <ArrowUp size={13} />
                          ) : (
                            <ArrowDown size={13} />
                          )}
                        </button>
                      </th>
                      {filters.view === "ranking" ? (
                        <>
                          <th scope="col" className="coverage-cell">
                            覆盖率
                          </th>
                          <th scope="col" className="wins-cell">
                            最优数据集
                          </th>
                          <th scope="col" className="action-cell">
                            <span className="sr-only">查看详情</span>
                          </th>
                        </>
                      ) : (
                        filters.datasets.map((id) => (
                          <th scope="col" key={id}>
                            {id}
                          </th>
                        ))
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {displayed.map((row) => (
                      <tr
                        key={row.id}
                        className={`${row.rank === 1 ? "first-place" : ""} ${selected.includes(row.id) ? "row-selected" : ""}`}
                      >
                        <td className="checkbox-cell">
                          <input
                            type="checkbox"
                            checked={selected.includes(row.id)}
                            aria-label={`对比 ${row.name}`}
                            onChange={() => toggleSelected(row.id)}
                          />
                        </td>
                        <td className="rank-cell">
                          <span className={`rank-number rank-${row.rank}`}>
                            {row.rank === null
                              ? "—"
                              : String(row.rank).padStart(2, "0")}
                          </span>
                        </td>
                        <td className="method-cell">
                          <button
                            className="method-button"
                            onClick={() => setDetailId(row.id)}
                          >
                            <MethodIcon row={row} />
                            <span>
                              <strong>
                                {row.name}
                                {row.rank === 1 && (
                                  <span className="best-tag">TOP</span>
                                )}
                              </strong>
                              <small>{row.family}</small>
                            </span>
                          </button>
                        </td>
                        <td className="score-cell">
                          <span
                            className={
                              row.score === null
                                ? "missing-score"
                                : "score-number"
                            }
                          >
                            {score(row.score)}
                          </span>
                          {filters.view === "ranking" && row.score !== null && (
                            <span className="score-track">
                              <span style={{ width: `${row.score * 100}%` }} />
                            </span>
                          )}
                        </td>
                        {filters.view === "ranking" ? (
                          <>
                            <td className="coverage-cell">
                              <span
                                className={
                                  row.coverage === filters.datasets.length
                                    ? "coverage complete"
                                    : "coverage"
                                }
                              >
                                {row.coverage === filters.datasets.length && (
                                  <Check size={11} />
                                )}
                                {row.coverage}/{filters.datasets.length}
                              </span>
                            </td>
                            <td className="wins-cell">
                              {row.wins > 0 ? (
                                <span className="wins">
                                  <Trophy size={12} />
                                  {row.wins}
                                </span>
                              ) : (
                                <span className="muted">—</span>
                              )}
                            </td>
                            <td className="action-cell">
                              <button
                                className="icon-button row-arrow"
                                aria-label={`查看 ${row.name} 详情`}
                                onClick={() => setDetailId(row.id)}
                              >
                                <ArrowUpRight size={16} />
                              </button>
                            </td>
                          </>
                        ) : (
                          filters.datasets.map((id) => {
                            const value = row.values[id];
                            const best = Math.max(
                              ...rows.map((r) => r.values[id] ?? -1),
                            );
                            return (
                              <td key={id}>
                                <span
                                  className={`heat-cell ${value !== null && value === best ? "cell-best" : ""}`}
                                  style={
                                    value === null
                                      ? undefined
                                      : ({
                                          "--heat-alpha": Math.max(
                                            0.03,
                                            (value - 0.4) * 0.38,
                                          ),
                                        } as CSSProperties)
                                  }
                                >
                                  {score(value)}
                                  {value !== null && value === best && (
                                    <span className="cell-dot" />
                                  )}
                                </span>
                              </td>
                            );
                          })
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {!displayed.length && (
                <div className="empty-state">
                  <Search size={26} />
                  <h3>没有找到匹配的方法</h3>
                  <p>试试其他关键词，或调整当前筛选条件。</p>
                  <button className="outline-button" onClick={resetFilters}>
                    重置筛选
                  </button>
                </div>
              )}
              <div className="table-footer">
                <span>
                  显示 {displayed.length} / {rows.length} 个方法
                </span>
                <span>
                  {filters.view === "ranking"
                    ? "勾选方法，进行并排对比"
                    : "● 金色标记表示该数据集的最优结果"}
                </span>
              </div>
            </section>

            {filters.view === "ranking" && (
              <aside className="insights">
                <section className="leader-card">
                  <div className="insight-label">
                    <Sparkles size={14} />
                    当前领先<span>{data.demo ? "示例分析" : "快照分析"}</span>
                  </div>
                  {leader ? (
                    <>
                      <div className="leader-name">
                        <MethodIcon row={leader} />
                        <h3>{leader.name}</h3>
                      </div>
                      <div className="leader-score">
                        {score(leader.score)}
                        <span>{metrics[filters.metric].short}</span>
                      </div>
                      <div className="leader-rule" />
                      <p>
                        在所选 {filters.datasets.length} 个数据集上的宏平均表现
                      </p>
                      <button onClick={() => setDetailId(leader.id)}>
                        查看方法详情 <ArrowRight size={15} />
                      </button>
                    </>
                  ) : (
                    <div className="insight-empty">
                      暂无完整覆盖所选数据集的方法
                    </div>
                  )}
                </section>
                <section className="dataset-leaders">
                  <h3>
                    数据集最佳表现
                    <button
                      className="help-button"
                      onClick={() => setModal("methodology")}
                      aria-label="了解数据集最佳表现"
                    >
                      <Info size={14} />
                    </button>
                  </h3>
                  <p>每个场景，都有自己的优等生。</p>
                  {filters.datasets.map((id) => {
                    const best = [...rows]
                      .filter((r) => r.values[id] !== null)
                      .sort(
                        (a, b) => (b.values[id] ?? -1) - (a.values[id] ?? -1),
                      )[0];
                    return (
                      <button
                        className="dataset-leader-row"
                        key={id}
                        onClick={() => best && setDetailId(best.id)}
                        disabled={!best}
                      >
                        <span>
                          <strong>{id}</strong>
                          <small>
                            {best?.name ?? "暂无结果"}
                            {best &&
                            rows.filter((r) => r.values[id] === best.values[id])
                              .length > 1
                              ? " · 并列"
                              : ""}
                          </small>
                        </span>
                        <span className="dataset-best-score">
                          {score(best?.values[id])}
                          <ArrowUpRight size={12} />
                        </span>
                      </button>
                    );
                  })}
                </section>
                <button
                  className="methodology-link"
                  onClick={() => setModal("methodology")}
                >
                  <FlaskConical size={16} />
                  <span>
                    排名是如何计算的？<small>了解评测协议与汇总规则</small>
                  </span>
                  <ArrowUpRight size={15} />
                </button>
              </aside>
            )}
          </div>

          <div className="benchmark-note">
            <Info size={15} />
            <p>
              {data.demo ? (
                <>
                  <strong>演示数据</strong> ·
                  当前分数用于展示界面与交互，不代表真实实验结论或方法兼容性。
                </>
              ) : (
                <>
                  <strong>本地导入结果</strong> ·
                  数据由当前会话导入，未经过平台审核，刷新后恢复演示数据。
                </>
              )}{" "}
              综合分数为数据集等权均值；{complete}/{rows.length}{" "}
              个方法完整覆盖，缺失值不计为 0。
            </p>
            <button onClick={() => setModal("import")}>
              导入结果 <Upload size={13} />
            </button>
          </div>
          <footer className="page-footer">
            <span>Built for better time series research.</span>
            <div>
              <span>EasyTSAD · 开放评测</span>
              <a
                href={`${repoUrl}/blob/main/LICENSE`}
                target="_blank"
                rel="noreferrer"
              >
                GPL-3.0 <ArrowUpRight size={11} />
              </a>
            </div>
          </footer>
        </main>
      </div>

      {selected.length > 0 && (
        <div className="compare-tray">
          <span className="compare-count">
            <GitCompareArrows size={18} />
            已选 {selected.length}/4
          </span>
          <div className="selected-methods">
            {compared.map((row) => (
              <button
                key={row.id}
                onClick={() => toggleSelected(row.id)}
                aria-label={`移除 ${row.name}`}
              >
                <span style={{ background: row.color }} />
                {row.name}
                <X size={12} />
              </button>
            ))}
          </div>
          <button className="text-button" onClick={() => setSelected([])}>
            清空
          </button>
          <button
            className="primary-button"
            onClick={() => setModal("compare")}
            disabled={selected.length < 2}
          >
            对比方法 <ArrowRight size={15} />
          </button>
        </div>
      )}
      {toast && (
        <div className="toast" role="status">
          <CheckCheck size={17} />
          {toast}
        </div>
      )}

      {modal === "methodology" && (
        <Dialog title="让每一个排名都有依据" onClose={() => setModal(null)}>
          <div className="methodology-intro">
            <FlaskConical size={22} />
            <p>相同任务、相同训练范式、相同指标与数据集范围，是比较的前提。</p>
          </div>
          <div className="explanation">
            {data.demo && (
              <div className="data-origin">
                <h3>当前数据来自哪里？</h3>
                <p>
                  当前分数是设计阶段生成的合成示例，用于验证排名、筛选、缺失值与对比交互。方法和数据集名称对应研究对象，但示例分数并非训练测得，也不是引用的论文成绩。
                </p>
                <p>
                  只有导入实际实验产出的评测文件后，这个会话才会展示真实结果。当前演示排名不能作为算法优劣的实验结论。
                </p>
              </div>
            )}
            <h3>01 / 指标与协议</h3>
            {Object.entries(metrics).map(([id, m]) => (
              <div key={id} className="metric-explanation">
                <strong>{m.label}</strong>
                <p>{m.description}</p>
              </div>
            ))}
            <h3>02 / 排名与缺失值</h3>
            <p>
              先读取每个数据集的
              avg.json（数据集内的序列均值），再对所选数据集做等权宏平均；按完整精度降序排名，同分采用竞赛排名（1、1、3）。页面展示
              4 位小数，显示值相同不一定代表原始值相同。
            </p>
            <p>
              只有完整覆盖所选数据集的方法参与综合排名。缺失值显示为「—」，不填零，不对部分结果计算综合均值。搜索和方法类型仅筛选展示内容，保留原始排名；切换数据集或指标会重新计算排名。最优数据集数量统计所有有结果的方法，允许并列。
            </p>
            <h3>03 / 训练与可复现性</h3>
            <p>
              MTS 对应仓库的多变量训练流程；单变量分别展示
              Naive、All-in-one、Zero-shot 与跨数据集
              Zero-shot。不同训练范式不混排。真实发布还需记录数据版本、预处理、随机种子、代码版本、超参数、评测
              margin 和协议参数。
            </p>
            <div className="inline-notice">
              <Info size={16} />
              <span>
                当前
                {data.demo
                  ? "为设计演示，所有分数都是合成数据。"
                  : "为本地导入快照，未经过平台审核。"}{" "}
                不展示未经多次实验计算的置信区间。
              </span>
            </div>
            <a
              className="outline-button inline-flex"
              href={`${docsUrl}API/Protocols/`}
              target="_blank"
              rel="noreferrer"
            >
              查看评测协议文档 <ExternalLink size={14} />
            </a>
          </div>
        </Dialog>
      )}

      {modal === "datasets" && (
        <Dialog title="选择你的评测场景" onClose={() => setModal(null)} wide>
          <p className="dialog-subtitle">
            当前快照包含 {data.datasets.length}{" "}
            个数据集。点击数据集，查看该场景下的方法表现。
          </p>
          <div className="dataset-catalog">
            {data.datasets.map((d) => (
              <button
                key={d.id}
                onClick={() => {
                  update({
                    task: d.task,
                    schema: d.task === "mts" ? "mts" : "naive",
                    datasets: [d.id],
                    search: "",
                    family: "all",
                  });
                  setSelected([]);
                  setModal(null);
                }}
              >
                <span className="catalog-icon">
                  <Database size={20} />
                </span>
                <span>
                  <strong>{d.id}</strong>
                  <small>{d.domain}</small>
                </span>
                <span className="catalog-task">{d.task.toUpperCase()}</span>
                <ArrowUpRight size={16} />
              </button>
            ))}
          </div>
        </Dialog>
      )}

      {modal === "compare" && (
        <Dialog title="方法并排对比" onClose={() => setModal(null)} wide>
          <p className="dialog-subtitle">
            {taskNames[filters.task]} · {schemas[filters.schema]} ·{" "}
            {metrics[filters.metric].label} ·{" "}
            {data.demo ? "演示数据" : "导入结果"}
          </p>
          <div className="compare-legend">
            {compared.map((row) => (
              <span key={row.id}>
                <i style={{ background: row.color }} />
                {row.name}
              </span>
            ))}
          </div>
          <div className="comparison-chart">
            {filters.datasets.map((id) => (
              <div className="comparison-group" key={id}>
                <strong>{id}</strong>
                <div>
                  {compared.map((row) => (
                    <div className="comparison-bar-row" key={row.id}>
                      <span className="sr-only">{row.name}</span>
                      <div className="comparison-track">
                        <div
                          style={{
                            width: `${(row.values[id] ?? 0) * 100}%`,
                            background: row.color,
                          }}
                        />
                      </div>
                      <span>{score(row.values[id])}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="comparison-summary">
            {compared.map((row) => (
              <div key={row.id}>
                <MethodIcon row={row} small />
                <strong>{row.name}</strong>
                <span>综合均值</span>
                <b>{score(row.score)}</b>
                <small>
                  覆盖 {row.coverage}/{filters.datasets.length} 个数据集
                </small>
              </div>
            ))}
          </div>
          <p className="dialog-footnote">
            横轴固定为
            0–1。缺失结果显示为「—」。颜色仅区分方法，不代表统计显著性。
          </p>
        </Dialog>
      )}

      {detail && (
        <Dialog title={detail.name} onClose={() => setDetailId(null)} wide>
          <div className="detail-top">
            <MethodIcon row={detail} />
            <div>
              <strong>{detail.description}</strong>
              <p>
                {detail.family} · {schemas[filters.schema]} ·{" "}
                {data.demo ? "演示结果" : "导入结果"}
              </p>
            </div>
            <a
              className="outline-button"
              href={`${repoUrl}/tree/main/EasyTSAD/Methods/${encodeURIComponent(detail.id)}`}
              target="_blank"
              rel="noreferrer"
            >
              <Code2 size={15} />
              方法源码
              <ArrowUpRight size={13} />
            </a>
          </div>
          <div className="detail-stats">
            <div>
              <span>当前综合排名</span>
              <strong>{detail.rank ? `#${detail.rank}` : "未排名"}</strong>
            </div>
            <div>
              <span>{metrics[filters.metric].label}</span>
              <strong>{score(detail.score)}</strong>
            </div>
            <div>
              <span>数据集覆盖</span>
              <strong>
                {detail.coverage}/{filters.datasets.length}
              </strong>
            </div>
          </div>
          <h3 className="detail-table-title">各数据集完整指标</h3>
          <div className="table-scroll">
            <table className="detail-table">
              <thead>
                <tr>
                  <th scope="col">数据集</th>
                  {Object.values(metrics).map((m) => (
                    <th scope="col" key={m.label}>
                      {m.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filters.datasets.map((id) => {
                  const result = data.results.find(
                    (r) =>
                      r.method === detail.id &&
                      r.schema === filters.schema &&
                      r.dataset === id,
                  );
                  return (
                    <tr key={id}>
                      <td>{id}</td>
                      {Object.keys(metrics).map((m) => (
                        <td key={m}>{score(result?.metrics[m as Metric])}</td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="dialog-footnote">
            F1 PA、Event F1 与原始 AUPRC
            使用不同评测协议，不应将三者混合为一个分数。
          </p>
        </Dialog>
      )}

      {modal === "import" && (
        <Dialog title="从实验结果到榜单" onClose={() => setModal(null)}>
          <p className="dialog-subtitle">
            载入 EasyTSAD benchmark
            JSON，使用相同的筛选、排名与对比视图探索你的结果。
          </p>
          <div className="import-area">
            <Upload size={28} />
            <h3>选择结果文件</h3>
            <p>JSON · 最大 10 MB · 文件仅在浏览器本地读取</p>
            <input
              ref={fileRef}
              type="file"
              accept=".json,application/json"
              aria-label="导入 benchmark JSON"
              onChange={(e) => void importFile(e.target.files?.[0])}
            />
            <button
              className="primary-button"
              onClick={() => fileRef.current?.click()}
            >
              选择 JSON 文件
            </button>
          </div>
          {importError && (
            <div role="alert" className="import-error">
              {importError}
            </div>
          )}
          <div className="explanation">
            <h3>从仓库导出</h3>
            <p>
              使用 leaderboard/scripts/export_results.py 将 Results/Evals 下的
              avg.json 转成此页面的数据格式。说明与字段定义见
              leaderboard/README.md。
            </p>
            <pre>
              python3 leaderboard/scripts/export_results.py \<br /> --results
              Results/Evals \<br /> --output benchmark.json
            </pre>
            <p>
              导入仅在当前会话有效。正式发布前，应补充实验配置和结果审核流程。
            </p>
          </div>
        </Dialog>
      )}
    </div>
  );
}
