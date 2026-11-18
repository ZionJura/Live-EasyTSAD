import { join } from 'node:path';

const plugin = (id, name, config) => ({id, name: `@deepseek-ai/dsh-${name}`, ...(config ? {config} : {})});
const roleText = {
  lead: 'Keep ordinary conversation natural and persistent. For explicit TSAD reproduction requests use tsad_* tools; the controller owns FETCH -> RESEARCH (hyperparameters, interface, environment in parallel) -> IMPLEMENT -> EVALUATE -> REPORT -> DONE, with DIAGNOSE only for concrete execution failures. Workflow workers use distinct CLI bindings. Use Goal and Todo where useful; tsad_team maintains the TSAD-local durable Team task board. This board is advisory and does not schedule experiments. Cite actual processes, progress and canonical avg.json/all.json artifacts for every execution or score claim. Never start duplicate training.',
  researcher: 'Complete only the assigned FETCH or RESEARCH branch. Use fetch-repository for locating and downloading the paper and official repository; without an official repository return unsupported. Use research-hyperparameters, research-interface or research-environment for the assigned research branch and write its required artifacts. Research branches run in parallel and must not overwrite one another. Do not start training or change shared evaluation code.',
  implementer: 'Follow the supplied implementation or diagnosis prompt and the three research artifacts. Adapt the assigned official implementation to EasyTSAD under the method write lease. Limit edits to the assigned method and environment. The controller owns training and evaluation. Report exact files, observed checks and unresolved differences; do not invent a workflow skill or a smoke gate.',
  supervisor: 'Diagnose only the concrete execution failure supplied by the controller. Read process, log, source and canonical score evidence; distinguish environment, interface, algorithm and resource causes. Do not infer failure from a low score or alter shared metrics or labels. Follow the fixed diagnosis prompt and return evidence and the narrow repair scope.',
  reporter: 'Read canonical avg.json and all.json references and their data. Summarize actual completion, parameters, processes and evaluation evidence when requested. The controller produces REPORT and DONE; no separate reporting skill or mandatory Markdown report is required.',
};

export function buildPatch(env, hostDir) {
  const provider = env.TSAD_MODEL_PROVIDER || 'terra';
  const model = env.TSAD_MODEL || 'deepseek-v4.1-flash';
  const modelInfo = {id: model, name: model, input: ['text'], contextWindow: Number(env.TSAD_MODEL_CONTEXT_WINDOW || 131072), maxTokens: Number(env.TSAD_MODEL_MAX_TOKENS || 16384)};
  const disabled = ['session-log-deepseek', 'plugin-package-inventory-deepseek', 'otel', 'session-telemetry-otel', 'command-feedback', 'desktop-product-telemetry', 'product-analytics',
    'llm-deepseek', 'llm-deepseek-account', 'deepseek-llm-api-extensions', 'session-title-llm', 'web-search-deepseek', 'web-deepseek-api', 'ptc-runtime', 'workflow-ptc', 'tool-workflow', 'hmr', 'plugin-manager',
    // Presets own these agent-scoped capabilities; no process-wide duplicates.
    'tool-plugin-manager', 'tool-bash', 'tool-pwsh', 'tool-jobs', 'tool-fs', 'tool-fs-search',
    'skill-filesystem', 'tool-skill', 'command-goal', 'tool-goal', 'plan-mode', 'compaction-basic',
    'command-compact', 'tool-result-pruner', 'tool-subagent-control', 'tool-subagent',
    'tool-subagent-fork', 'agent-instructions', 'tool-todo', 'tool-web'];
  const patch = disabled.map(id => ({id, disabled: true}));
  patch.push({id: 'llm-pi-ai', config: {providers: {[provider]: {
    displayName: 'TSAD model', apiKeyEnv: env.TSAD_MODEL_API_KEY_ENV || 'TERRA_PROXY_API_KEY', api: env.TSAD_MODEL_API || 'openai-completions',
    baseURL: env.TSAD_MODEL_BASE_URL || 'http://127.0.0.1:8080/v1', models: [modelInfo],
    retryPolicy: {mode: 'normal', maxRetries: 3, retryableCodes: ['EMPTY_RESPONSE', 'RATE_LIMIT', 'SERVER', 'TIMEOUT', 'TRANSPORT', 'PI_AI_ERROR']},
  }}}});
  patch.push({id: 'agent-default-model', config: {provider, model}});
  patch.push({id: 'subagent', config: {maxDepth: 2, maxActiveSubagents: 8}});
  patch.push({id: 'web', config: {searchProvider: 'tsad-searxng', fetchProvider: 'http'}});
  // Compose only the native backend services used by the TUI bridge. Connection
  // also works without a webserver (the upstream SDK uses this composition).
  const hostServices = [plugin('connection', 'client-connection'), plugin('file-upload', 'client-file-upload'),
    plugin('workspace', 'workspace'), plugin('file-reference-local', 'file-reference-local'),
    plugin('session-controller', 'api-session-controller', {nativeOpen: false}),
    plugin('agent-preset-registry', 'agent-preset-registry', {default: 'tsad-lead'})];
  patch.push({id: 'system-prompt', config: {personaPrefix: 'You are a coding agent powered by the {{model}} model.'}});
  const presets = Object.entries(roleText).map(([role, prompt], index) => {
    const write = role === 'implementer';
    const children = [plugin('persona', 'persona', {prefix: `${prompt}\nYou are a TSAD ${role} assistant powered by {{model}}. Prefer concise Chinese unless the user chooses another language.`}),
      plugin('tool-fs', 'tool-fs', {read: true, readImage: true, write, edit: write, applyPatch: false}),
      plugin('tool-fs-search', 'tool-fs-search', {sampleOverCapGlobResults: false}),
      plugin('skill-filesystem', 'skill-filesystem', {includeDefaultRoots: false, customSkillDirs: [join(hostDir, 'skills')], watch: false}),
      plugin('tool-skill', 'tool-skill'), plugin('tool-todo', 'tool-todo', {allowParallelInProgress: true}),
      plugin('tool-web', 'tool-web', {fetch: true, searchTimeoutMs: 60000}), plugin('compaction-basic', 'compaction-basic'), plugin('command-compact', 'command-compact')];
    if (role === 'lead') children.push(plugin('command-goal', 'command-goal'), plugin('tool-goal', 'tool-goal'), plugin('plan-mode', 'plan-mode', {
      section: 'You are in plan mode. Inspect source and evidence, clarify requirements and design the requested work. Use read-only research tools. Do not modify files, create experiments, install dependencies or start training until the user approves the plan. Present the complete plan with exit_plan_mode; after approval, continue the authorized work. The TSAD controller remains responsible for experiment stages and execution.',
    }),
      plugin('tool-subagent-control', 'tool-subagent-control'), plugin('tool-subagent', 'tool-subagent', {provider: 'spawn', toolName: 'subagent'}),
      plugin('tool-subagent-fork', 'tool-subagent', {provider: 'fork', toolName: 'subagent_fork'}));
    return plugin(`tsad-preset-${role}`, 'agent-preset', {id: `tsad-${role}`, order: index, plugins: [{
      id: `tsad-${role}-composition`, name: 'cordis:group', group: true,
      isolate: {compaction: true, planMode: true}, config: children,
    }]});
  });
  patch.push({insert: [...hostServices, ...presets, {id: 'tsad-native-bridge', name: join(hostDir, 'src/plugin.mjs')}]});
  return patch;
}
