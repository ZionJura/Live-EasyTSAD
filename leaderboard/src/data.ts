export type Task = "mts" | "uts";
export type Schema =
  | "mts"
  | "naive"
  | "all_in_one"
  | "zero_shot"
  | "zero_shot_cross_ds";
export type Metric = "event_f1" | "point_f1" | "auprc";
export type Family = "深度学习" | "传统方法" | "预训练模型";
export interface Method {
  id: string;
  name: string;
  family: Family;
  description: string;
  color: string;
}
export interface Dataset {
  id: string;
  task: Task;
  domain: string;
}
export interface Result {
  method: string;
  dataset: string;
  schema: Schema;
  metrics: Partial<Record<Metric, number | null>>;
}
export interface Benchmark {
  version: 1;
  demo: boolean;
  updatedAt: string;
  methods: Method[];
  datasets: Dataset[];
  results: Result[];
}

export const metrics: Record<
  Metric,
  { label: string; short: string; description: string }
> = {
  event_f1: {
    label: "Event F1 · Log",
    short: "Event F1",
    description:
      'EventF1PA(mode="log", base=3)：按事件长度的对数加权，使用 point adjustment。分数范围 0–1，越高越好。',
  },
  point_f1: {
    label: "Point F1 · PA",
    short: "Point F1",
    description:
      "PointF1PA：使用 point adjustment 的最佳逐点 F1。与未调整的 F1 不可直接比较。分数范围 0–1，越高越好。",
  },
  auprc: {
    label: "AUPRC · Point",
    short: "AUPRC",
    description:
      "PointPrc：原始逐点 average precision，不使用 point adjustment。分数范围 0–1，越高越好。",
  },
};
export const schemas: Record<Schema, string> = {
  mts: "MTS · 多变量训练",
  naive: "Naive · 独立训练",
  all_in_one: "All-in-one · 联合训练",
  zero_shot: "Zero-shot · 跨序列",
  zero_shot_cross_ds: "Zero-shot · 跨数据集",
};
export const repoUrl = "https://github.com/ZionJura/Live-EasyTSAD";
export const docsUrl = "https://dawnvince.github.io/EasyTSAD/";

const methods: Method[] = [
  {
    id: "DCdetector",
    name: "DCdetector",
    family: "深度学习",
    description: "双重注意力对比学习",
    color: "#3467ab",
  },
  {
    id: "AnomalyTransformer",
    name: "Anomaly Transformer",
    family: "深度学习",
    description: "基于关联差异的异常检测",
    color: "#6b86b2",
  },
  {
    id: "TranAD",
    name: "TranAD",
    family: "深度学习",
    description: "Transformer 异常检测",
    color: "#b48a38",
  },
  {
    id: "TimesNet",
    name: "TimesNet",
    family: "深度学习",
    description: "时序二维变化建模",
    color: "#769dc7",
  },
  {
    id: "FITS",
    name: "FITS",
    family: "深度学习",
    description: "频域插值时序建模",
    color: "#526e9c",
  },
  {
    id: "OFA",
    name: "OFA",
    family: "预训练模型",
    description: "One Fits All · 预训练语言模型",
    color: "#386b8f",
  },
  {
    id: "AE",
    name: "AutoEncoder",
    family: "深度学习",
    description: "自编码器重构基线",
    color: "#bd9b56",
  },
  {
    id: "EncDecAD",
    name: "EncDecAD",
    family: "深度学习",
    description: "LSTM 编码器与解码器",
    color: "#8a9db8",
  },
  {
    id: "OCSVM",
    name: "OCSVM",
    family: "传统方法",
    description: "单类支持向量机",
    color: "#587a9e",
  },
  {
    id: "LOF",
    name: "LOF",
    family: "传统方法",
    description: "局部离群因子",
    color: "#a08b59",
  },
];
const datasets: Dataset[] = [
  { id: "SMD", task: "mts", domain: "服务器监控" },
  { id: "MSL", task: "mts", domain: "航天器遥测" },
  { id: "SMAP", task: "mts", domain: "卫星遥测" },
  { id: "SWaT", task: "mts", domain: "工业水处理" },
  { id: "PSM", task: "mts", domain: "服务器监控" },
  { id: "WADI", task: "mts", domain: "供水系统" },
  { id: "AIOPS", task: "uts", domain: "运维监控" },
  { id: "NAB", task: "uts", domain: "流式监控" },
  { id: "TODS", task: "uts", domain: "综合时序" },
  { id: "WSD", task: "uts", domain: "网络服务" },
  { id: "Yahoo", task: "uts", domain: "服务流量" },
  { id: "UCR", task: "uts", domain: "综合时序" },
];

// Deterministic design fixtures, NOT experiment measurements or compatibility claims.
// Independent variations deliberately exercise different metric/dataset rankings.
const bases = [
  0.917, 0.892, 0.871, 0.854, 0.831, 0.813, 0.766, 0.742, 0.683, 0.641,
];
function variation(key: string) {
  let hash = 0;
  for (const char of key) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return ((hash % 151) - 75) / 1000;
}
const results: Result[] = datasets.flatMap((dataset) =>
  (dataset.task === "mts"
    ? (["mts"] as Schema[])
    : (["naive", "all_in_one", "zero_shot", "zero_shot_cross_ds"] as Schema[])
  ).flatMap((schema) =>
    methods.map((method, i) => ({
      method: method.id,
      dataset: dataset.id,
      schema,
      metrics: Object.fromEntries(
        Object.keys(metrics).map((metric) => [
          metric,
          (i === 5 && dataset.id === "WADI") ||
          (i === 9 && dataset.id === "UCR")
            ? null
            : Number(
                Math.min(
                  0.992,
                  Math.max(
                    0,
                    bases[i] +
                      variation(
                        `${method.id}:${dataset.id}:${schema}:${metric}`,
                      ),
                  ),
                ).toFixed(4),
              ),
        ]),
      ),
    })),
  ),
);
export const demoBenchmark: Benchmark = {
  version: 1,
  demo: true,
  updatedAt: "2026-11-04",
  methods,
  datasets,
  results,
};

export interface RankedMethod extends Method {
  values: Record<string, number | null>;
  score: number | null;
  coverage: number;
  wins: number;
  rank: number | null;
}

export function rankMethods(
  data: Benchmark,
  schema: Schema,
  metric: Metric,
  datasetIds: string[],
): RankedMethod[] {
  const records = data.results.filter((row) => row.schema === schema);
  const rows: RankedMethod[] = data.methods
    .filter((method) => records.some((row) => row.method === method.id))
    .map((method) => {
      const values = Object.fromEntries(
        datasetIds.map((id) => [
          id,
          records.find((row) => row.method === method.id && row.dataset === id)
            ?.metrics[metric] ?? null,
        ]),
      );
      const available = Object.values(values).filter(
        (v): v is number => v !== null && Number.isFinite(v),
      );
      return {
        ...method,
        values,
        score:
          datasetIds.length > 0 && available.length === datasetIds.length
            ? available.reduce((a, b) => a + b, 0) / available.length
            : null,
        coverage: available.length,
        wins: 0,
        rank: null,
      };
    });
  for (const id of datasetIds) {
    const best = Math.max(...rows.map((row) => row.values[id] ?? -Infinity));
    for (const row of rows) if (row.values[id] === best) row.wins++;
  }
  rows.sort(
    (a, b) => (b.score ?? -1) - (a.score ?? -1) || a.name.localeCompare(b.name),
  );
  rows.forEach((row, i) => {
    if (row.score !== null)
      row.rank =
        i > 0 && Math.abs(row.score - (rows[i - 1].score ?? -1)) < 1e-12
          ? rows[i - 1].rank
          : i + 1;
  });
  return rows;
}

export function validateBenchmark(value: unknown): Benchmark {
  if (!value || typeof value !== "object")
    throw new Error("文件必须是 JSON 对象。");
  const data = value as Benchmark;
  if (
    data.version !== 1 ||
    typeof data.demo !== "boolean" ||
    !/^\d{4}-\d{2}-\d{2}$/.test(data.updatedAt) ||
    !Array.isArray(data.methods) ||
    !Array.isArray(data.datasets) ||
    !Array.isArray(data.results)
  )
    throw new Error("数据格式不正确，请使用 version 1 的 benchmark JSON。");
  if (!data.methods.length || !data.datasets.length || !data.results.length)
    throw new Error("方法、数据集和结果不能为空。");
  const methodIds = new Set<string>();
  for (const m of data.methods) {
    if (
      !m ||
      typeof m.id !== "string" ||
      !/^[\w.-]+$/.test(m.id) ||
      methodIds.has(m.id) ||
      typeof m.name !== "string" ||
      typeof m.description !== "string" ||
      !["深度学习", "传统方法", "预训练模型"].includes(m.family) ||
      !/^#[\da-f]{6}$/i.test(m.color)
    )
      throw new Error("方法元数据不合法或 ID 重复。");
    methodIds.add(m.id);
  }
  const datasetIds = new Map<string, Task>();
  for (const d of data.datasets) {
    if (
      !d ||
      typeof d.id !== "string" ||
      !/^[\w.-]+$/.test(d.id) ||
      datasetIds.has(d.id) ||
      !["mts", "uts"].includes(d.task) ||
      typeof d.domain !== "string"
    )
      throw new Error("数据集元数据不合法或 ID 重复。");
    datasetIds.set(d.id, d.task);
  }
  const keys = new Set<string>();
  for (const r of data.results) {
    if (
      !r ||
      !methodIds.has(r.method) ||
      !datasetIds.has(r.dataset) ||
      !Object.hasOwn(schemas, r.schema) ||
      (r.schema === "mts") !== (datasetIds.get(r.dataset) === "mts") ||
      !r.metrics ||
      typeof r.metrics !== "object"
    )
      throw new Error("结果引用的方法、任务、数据集或训练范式不合法。");
    const key = `${r.method}/${r.schema}/${r.dataset}`;
    if (keys.has(key)) throw new Error("存在重复的结果记录。");
    keys.add(key);
    for (const [metric, score] of Object.entries(r.metrics))
      if (
        !Object.hasOwn(metrics, metric) ||
        (score !== null &&
          (typeof score !== "number" ||
            !Number.isFinite(score) ||
            score < 0 ||
            score > 1))
      )
        throw new Error("指标值必须为 0–1 的有限数值或 null。");
  }
  return data;
}

export function resultsCsv(
  rows: RankedMethod[],
  datasetIds: string[],
  metric: Metric,
  schema: Schema,
  demo: boolean,
) {
  const escape = (value: unknown) => {
    const s = String(value ?? "");
    return (
      '"' + (/^[=+@\-\t\r]/.test(s) ? "'" : "") + s.replaceAll('"', '""') + '"'
    );
  };
  return (
    "\uFEFF" +
    [
      [
        "data_source",
        "schema",
        "metric",
        "rank",
        "method",
        "macro_average",
        "coverage",
        ...datasetIds,
      ],
      ...rows.map((row) => [
        demo ? "DEMO" : "IMPORTED",
        schema,
        metric,
        row.rank,
        row.name,
        row.score?.toFixed(6),
        `${row.coverage}/${datasetIds.length}`,
        ...datasetIds.map((id) => row.values[id]),
      ]),
    ]
      .map((row) => row.map(escape).join(","))
      .join("\r\n")
  );
}
