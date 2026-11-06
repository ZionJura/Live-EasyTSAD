import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import {
  demoBenchmark,
  rankMethods,
  validateBenchmark,
  resultsCsv,
} from "../src/data";
import type { Benchmark } from "../src/data";

const fixture: Benchmark = {
  version: 1,
  demo: false,
  updatedAt: "2026-11-04",
  methods: ["A", "B", "C", "D"].map((id) => ({
    id,
    name: `Method ${id}`,
    family: "深度学习",
    description: "Fixture",
    color: "#738f78",
  })),
  datasets: ["SMD", "MSL"].map((id) => ({
    id,
    task: "mts",
    domain: "Fixture",
  })),
  results: [
    {
      method: "A",
      dataset: "SMD",
      schema: "mts",
      metrics: { event_f1: 0.9, auprc: 0.3 },
    },
    {
      method: "A",
      dataset: "MSL",
      schema: "mts",
      metrics: { event_f1: 0.7, auprc: 0.3 },
    },
    {
      method: "B",
      dataset: "SMD",
      schema: "mts",
      metrics: { event_f1: 0.8, auprc: 0.9 },
    },
    {
      method: "B",
      dataset: "MSL",
      schema: "mts",
      metrics: { event_f1: 0.8, auprc: 0.9 },
    },
    {
      method: "C",
      dataset: "SMD",
      schema: "mts",
      metrics: { event_f1: 1, auprc: 1 },
    },
    { method: "D", dataset: "SMD", schema: "mts", metrics: { event_f1: 0.1 } },
    { method: "D", dataset: "MSL", schema: "mts", metrics: { event_f1: 0.1 } },
  ],
};

test("ranking uses equal dataset weights, competition ties and complete coverage", () => {
  const rows = rankMethods(fixture, "mts", "event_f1", ["SMD", "MSL"]);
  expect(rows.map((row) => [row.id, row.rank])).toEqual([
    ["A", 1],
    ["B", 1],
    ["D", 3],
    ["C", null],
  ]);
  expect(rows[0].score).toBeCloseTo(0.8);
  expect(rows[3].score).toBeNull();
  expect(rows[3].wins).toBe(1);
  expect(rankMethods(fixture, "mts", "event_f1", ["SMD"])[0].id).toBe("C");
  expect(rankMethods(fixture, "mts", "auprc", ["SMD", "MSL"])[0].id).toBe("B");
  expect(rankMethods(fixture, "naive", "event_f1", ["SMD"])).toEqual([]);
});

test("imports reject bad references, non-finite scores, duplicates and incompatible schemas", () => {
  expect(validateBenchmark(fixture)).toBe(fixture);
  for (const mutate of [
    (data: Benchmark) => {
      data.results[0].metrics.event_f1 = Infinity;
    },
    (data: Benchmark) => {
      data.results[0].metrics.event_f1 = -1;
    },
    (data: Benchmark) => {
      data.results[0].method = "missing";
    },
    (data: Benchmark) => {
      data.results[0].schema = "naive";
    },
    (data: Benchmark) => {
      data.results.push(data.results[0]);
    },
    (data: Benchmark) => {
      data.methods.push(data.methods[0]);
    },
  ]) {
    const data = structuredClone(fixture);
    mutate(data);
    expect(() => validateBenchmark(data)).toThrow();
  }
  const rows = rankMethods(fixture, "mts", "event_f1", ["SMD", "MSL"]);
  rows[0].name = '=HYPERLINK("example")';
  expect(resultsCsv(rows, ["SMD", "MSL"], "event_f1", "mts", false)).toContain(
    "'=HYPERLINK",
  );
});

test("task, metric, search, category and matrix persist in a shared URL", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "时序异常检测榜单." }),
  ).toBeVisible();
  await page
    .getByLabel("检测任务")
    .getByRole("button", { name: "单变量异常检测 UTS", exact: true })
    .click();
  await page.getByLabel("训练范式").selectOption("all_in_one");
  await page.getByLabel("评测指标", { exact: true }).selectOption("auprc");
  await page.getByRole("button", { name: "结果矩阵", exact: true }).click();
  await page.getByLabel("搜索方法").fill("Times");
  await page.getByLabel("方法类型").selectOption("深度学习");
  await expect(page.locator("tbody tr")).toHaveCount(1);
  await expect(
    page.getByRole("columnheader", { name: "AIOPS", exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByLabel("训练范式")).toHaveValue("all_in_one");
  await expect(page.getByLabel("评测指标", { exact: true })).toHaveValue(
    "auprc",
  );
  await expect(page.getByLabel("搜索方法")).toHaveValue("Times");
  await expect(page.locator("tbody tr")).toHaveCount(1);
  expect(errors).toEqual([]);
});

test("dataset selection recalculates coverage and cannot become empty", async ({
  page,
}) => {
  await page.goto("/");
  const ofa = page
    .locator("tbody tr")
    .filter({
      has: page.getByRole("button", { name: "OFA 预训练模型", exact: true }),
    });
  await expect(ofa).toContainText("5/6");
  await page.getByRole("button", { name: "全部数据集" }).click();
  await page.locator("#dataset-picker").getByLabel("WADI").uncheck();
  await page.keyboard.press("Escape");
  await expect(ofa).toContainText("5/5");
  await expect(ofa.locator(".score-number")).toHaveCount(1);
  await page.getByRole("button", { name: "自定义数据集" }).click();
  for (const id of ["MSL", "SMAP", "SWaT", "PSM"])
    await page
      .locator("#dataset-picker")
      .getByLabel(id, { exact: false })
      .uncheck();
  await page.locator("#dataset-picker").getByLabel("SMD").click();
  await expect(page.locator("#dataset-picker input:checked")).toHaveCount(1);
  await expect(page.getByRole("status")).toContainText("至少保留一个");
});

test("comparison limits selection and dialog supports keyboard dismissal", async ({
  page,
}) => {
  await page.goto("/");
  const choices = page.locator("tbody input[type=checkbox]");
  for (let i = 0; i < 4; i++) await choices.nth(i).check();
  await choices.nth(4).click();
  await expect(page.locator("tbody input:checked")).toHaveCount(4);
  await page.getByRole("button", { name: "对比方法", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("方法并排对比");
  await expect(page.locator(".comparison-group")).toHaveCount(6);
  await expect(page.locator(".comparison-bar-row")).toHaveCount(24);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "对比方法", exact: true }),
  ).toBeFocused();
});

test("CSV export matches the filtered table and explicitly identifies demo data", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("搜索方法").fill("TranAD");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出 CSV" }).click();
  const file = await download;
  const csv = await readFile((await file.path())!, "utf8");
  expect(file.suggestedFilename()).toContain("demo-mts-mts-event_f1");
  expect(csv.split("\r\n")).toHaveLength(2);
  expect(csv).toContain('"DEMO","mts","event_f1"');
  expect(csv).toContain('"TranAD"');
  expect(csv).toContain('"SMD","MSL","SMAP","SWaT","PSM","WADI"');
});

test("real imports replace demo data, surface invalid files and recompute ranks", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "导入结果", exact: true }).click();
  await page
    .getByLabel("导入 benchmark JSON")
    .setInputFiles({
      name: "invalid.json",
      mimeType: "application/json",
      buffer: Buffer.from("{}"),
    });
  await expect(page.getByRole("alert")).toBeVisible();
  await page
    .getByLabel("导入 benchmark JSON")
    .setInputFiles({
      name: "results.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(fixture)),
    });
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator(".preview-tag")).toHaveText("已导入结果");
  await expect(page.locator("tbody tr")).toHaveCount(4);
  await expect(page.locator("tbody tr").first()).toContainText("0.8000");
  await expect(page.locator("tbody tr").last()).toContainText("1/2");
  await page.getByLabel("评测指标", { exact: true }).selectOption("auprc");
  await expect(page.locator("tbody tr").first()).toContainText("Method B");
});

test("search empty state resets, sorting works and detail shows independent protocols", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("搜索方法").fill("does-not-exist");
  await expect(
    page.getByRole("heading", { name: "没有找到匹配的方法" }),
  ).toBeVisible();
  await page
    .locator(".empty-state")
    .getByRole("button", { name: "重置筛选" })
    .click();
  await expect(page.locator("tbody tr")).toHaveCount(10);
  await page.getByRole("button", { name: "Event F1", exact: true }).click();
  const first = page.locator("tbody tr").first();
  await expect(first).toContainText("LOF");
  await first.getByRole("button", { name: "查看 LOF 详情" }).click();
  await expect(
    page
      .getByRole("dialog")
      .getByRole("columnheader", { name: "AUPRC · Point" }),
  ).toBeVisible();
  await expect(page.getByRole("dialog").locator("tbody tr")).toHaveCount(6);
});

test("mobile layout contains horizontal overflow and provides accessible navigation", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "打开导航" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "数据集", exact: true })
    .click();
  await page.getByRole("button", { name: "SMD 服务器监控 MTS" }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator(".coverage").first()).toHaveText("1/1");
  await page.getByRole("button", { name: "结果矩阵", exact: true }).click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
});

test("demo fixtures are valid but never marked as published measurements", () => {
  expect(validateBenchmark(demoBenchmark).demo).toBe(true);
});
