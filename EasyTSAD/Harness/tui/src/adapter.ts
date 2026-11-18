/** Remote-only terminal adapter. The controller remains the sole runtime owner. */
import {readFile} from 'node:fs/promises';
import {spawn, type ChildProcess} from 'node:child_process';
import {join, resolve} from 'node:path';
import {homedir} from 'node:os';
import {parse as parseToml} from 'smol-toml';
import {randomUUID} from 'node:crypto';

type Json = Record<string, unknown>;
export type RemoteBlock = Record<string, unknown>;
export interface RemoteMessage {
  id: string;
  role: string;
  content: string | RemoteBlock[];
  name?: string;
  toolName?: string;
  toolCallId?: string;
  streaming?: boolean;
  isError?: boolean;
  meta?: unknown;
}
export interface RemoteAdapterOptions {
  url?: string;
  token?: string;
  tokenFile?: string;
  root?: string;
  python?: string;
  refresh?: number;
  task?: string;
  resume?: string;
  gpus?: string;
  onMessages(messages: RemoteMessage[]): void;
  onStatus(text: string): void;
  onNotice(text: string): void;
}
const record = (value: unknown): Json => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Json : {};
const text = (value: unknown): string => typeof value === 'string' ? value : '';
const entries = (value: unknown): unknown[] => Array.isArray(value) ? value : [];
const sessionId = (value: unknown): string => text(record(value).sessionId || record(value).id);
const print = (value: unknown): string => typeof value === 'string' ? value : JSON.stringify(value, null, 2) ?? '';
const pathId = (id: string): string => encodeURIComponent(id);
const errorText = (cause: unknown): string => cause instanceof Error ? cause.message : String(cause);

/** Preserve model content blocks, call identities and order for native tool cards. */
export function normalizeHistory(snapshot: Json): RemoteMessage[] {
  if ((snapshot.transport === 'cli' || snapshot.eventEpoch) && Array.isArray(snapshot.events)) return cliHistory(snapshot);
  let source = entries(snapshot.history);
  if (!Array.isArray(snapshot.history)) source = entries(snapshot.messages);
  const page = record(snapshot.historyPage);
  if (!source.length && Array.isArray(page.records)) {
    const surface: Json[] = [];
    for (const row of page.records) {
      const envelope = record(row);
      const event = envelope.type === 'event' ? record(envelope.event) : envelope;
      const operation = record(event.surfaceOp);
      let at = surface.length;
      if (operation.op === 'replace') {
        const start = Number(operation.startSeq), end = Number(operation.endSeq);
        const position = surface.findIndex(item => Number(item.seq) >= start && Number(item.seq) <= end);
        if (position >= 0) at = position;
        for (let i = surface.length - 1; i >= 0; i--) if (Number(surface[i].seq) >= start && Number(surface[i].seq) <= end) surface.splice(i, 1);
      }
      const data = record(event.data);
      const message = event.type === 'user/message' ? data : record(data.message);
      if (message.role) surface.splice(at, 0, {...message, meta: message.meta ?? event.meta, seq: event.seq});
    }
    source = surface;
  }
  const result: RemoteMessage[] = [];
  for (const [index, value] of source.entries()) {
    const outer = record(value), message = record(outer.message || value);
    const role = text(message.role) || 'assistant';
    if (role === 'system' || ['runtime-context', 'skill-catalog', 'system-prompt'].includes(text(record(message.source).kind))) continue;
    const raw = message.content ?? message.text;
    const content = typeof raw === 'string' ? raw : entries(raw).filter(v => v !== null && typeof v === 'object') as RemoteBlock[];
    if (!content.length) continue;
    result.push({id: String(message.id ?? message.seq ?? outer.seq ?? `history-${index}`), role, content,
      ...(message.meta !== undefined || outer.meta !== undefined ? {meta: message.meta ?? outer.meta} : {}),
      ...(message.isError === true ? {isError: true} : {}),
      ...(message.name ? {name: text(message.name)} : {}),
      ...(message.toolName ? {toolName: text(message.toolName)} : {}),
      ...(message.callId || message.toolCallId || message.tool_call_id ? {toolCallId: text(message.callId || message.toolCallId || message.tool_call_id)} : {})});
  }
  return result;
}


/** Headless records are already committed step output, not token deltas. */
function cliHistory(snapshot: Json): RemoteMessage[] {
  const result: RemoteMessage[] = [];
  for (const [index, value] of entries(snapshot.messages).entries()) {
    const row = record(value), body = text(row.text || row.content);
    if (body) result.push({id: String(row.id || `cli-user-${index}`), role: text(row.role) || 'user', content: body});
  }
  for (const [index, value] of entries(snapshot.events).entries()) {
    const event = record(value), id = `cli-event-${index}`;
    if (event.type === 'text' && event.text) result.push({id, role: 'assistant', content: text(event.text)});
    else if (event.type === 'tool_call') result.push({id, role: 'assistant', content: [{type: 'tool-call', id: event.callId, name: event.tool, arguments: event.input ?? {}}]});
    else if (event.type === 'tool_result') result.push({id, role: 'tool', toolCallId: text(event.callId), content: [{type: 'text', text: print(event.result ?? event.error ?? '')}]});
    else if (event.type === 'final' && event.text) {
      const final = text(event.text);
      const tail = [...result].reverse().find(item => item.role === 'assistant' && typeof item.content === 'string');
      if (tail?.content !== final) result.push({id, role: 'assistant', content: final});
    }
  }
  const canonical = entries(snapshot.history).map(record).find(row => row.role === 'assistant');
  if (canonical) {
    const final = text(canonical.content || canonical.text);
    const tail = [...result].reverse().find(item => item.role === 'assistant' && typeof item.content === 'string');
    if (final && tail?.content !== final) result.push({id: 'cli-canonical-final', role: 'assistant', content: final});
  }
  return result;
}

/** Native and legacy workers publish different streaming envelopes. */
export function streamDelta(value: unknown): string {
  const data = record(value), frame = record(data.frame || data), chunk = record(frame.chunk || frame.event || frame);
  return ['text-delta', 'text_delta', 'text'].includes(text(chunk.type)) ? text(chunk.text ?? chunk.delta) : '';
}

export class RemoteHarnessAdapter {
  readonly options: RemoteAdapterOptions;
  selectedSessionId = '';
  snapshot: Json = {};
  private url: string;
  private token: string;
  private readonly explicitToken: boolean;
  private readonly explicitUrl: boolean;
  private readonly root: string;
  private readonly python: string;
  private tokenFile: string;
  private configDiscovered = false;
  private disposed = false;
  private generation = 0;
  private connectPromise?: Promise<void>;
  private polling = false;
  private timer?: ReturnType<typeof setTimeout>;
  private streamController?: AbortController;
  private requestControllers = new Set<AbortController>();
  private bootstrap?: ChildProcess;
  private bootstrapCancel?: () => void;
  private eventCursor = 0;
  private eventEpoch = '';
  private streamText = '';
  private cliFinalPending = false;
  private snapshotNeeded = true;
  private snapshotDeadline = 0;
  private selectedJob: string;
  private job: Json = {};
  private history: RemoteMessage[] = [];
  private notices: RemoteMessage[] = [];
  private historyRecords = new Map<number, Json>();
  private historyHasMore = false;
  private historyBefore: number | undefined;
  private published = '';
  private businessCursor = '';
  private details = false;
  private ready = false;
  private commandDescriptors: Json[] = [];
  private commandsNeeded = true;
  private metadataLoading = false;
  private selectionIntent = 0;
  private creations = new Map<string, Json>();

  constructor(options: RemoteAdapterOptions) {
    this.options = options;
    this.root = resolve(options.root || process.env.TSAD_ROOT || process.cwd());
    this.url = (options.url || process.env.TSAD_HARNESS_URL || 'http://127.0.0.1:8767').replace(/\/$/, '');
    this.explicitUrl = Boolean(options.url || process.env.TSAD_HARNESS_URL);
    this.explicitToken = Boolean(options.token || process.env.TSAD_HARNESS_TOKEN);
    this.token = options.token || process.env.TSAD_HARNESS_TOKEN || '';
    const runtime = process.env.TSAD_HARNESS_RUNTIME || 'runtime/harness-workbench';
    this.tokenFile = options.tokenFile || process.env.TSAD_HARNESS_TOKEN_FILE || resolve(this.root, runtime, 'access-token');
    this.python = options.python || process.env.TSAD_HARNESS_PYTHON || join(this.root, 'envs/harness/venv/bin/python');
    this.selectedJob = options.task || '';
  }

  getState(): Json { return {sessionId: this.selectedSessionId, snapshot: this.snapshot, job: this.job, connected: this.ready, commands: this.commandDescriptors, creations: [...this.creations.values()]}; }
  get commands(): Json[] { return this.commandDescriptors; }
  setDetails(enabled: boolean): void { this.details = enabled; this.emit(); }

  private alive(generation: number, id = this.selectedSessionId): boolean {
    return !this.disposed && generation === this.generation && id === this.selectedSessionId;
  }

  /** Read only terminal connection settings after the first rendered frame. */
  private async discoverConfig(): Promise<void> {
    if (this.configDiscovered) return;
    const explicitFile = Boolean(this.options.tokenFile || process.env.TSAD_HARNESS_TOKEN_FILE);
    if (this.explicitUrl && (this.explicitToken || explicitFile)) { this.configDiscovered = true; return; }
    const expand = (path: string) => path === '~' ? homedir() : path.startsWith('~/') ? join(homedir(), path.slice(2)) : path;
    const file = resolve(this.root, expand(process.env.TSAD_HARNESS_CONFIG || 'EasyTSAD/Harness/config.toml'));
    let config: Json;
    try { config = record(parseToml(await readFile(file, 'utf8'))); }
    catch (error) { throw new Error(`无法读取 Harness 配置 ${file}：${errorText(error)}`); }
    const interactive = record(config.interactive);
    if (!this.explicitUrl) {
      const port = Number(interactive.port ?? 8767);
      if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('interactive.port 必须是 1–65535 的整数');
      this.url = `http://127.0.0.1:${port}`;
    }
    if (!explicitFile) {
      const runtime = expand(process.env.TSAD_HARNESS_RUNTIME || text(interactive.runtime_dir) || 'runtime/harness-workbench');
      this.tokenFile = join(resolve(this.root, runtime), 'access-token');
    }
    this.configDiscovered = true;
  }

  private async readToken(): Promise<void> {
    if (this.explicitToken) return;
    try { this.token = (await readFile(this.tokenFile, 'utf8')).trim(); }
    catch (error) { if (record(error).code !== 'ENOENT') throw error; }
  }
  private async api(method: string, path: string, body?: unknown, timeout = method === 'GET' ? 15_000 : 120_000): Promise<Json> {
    if (this.disposed) throw new Error('Terminal client is closed');
    const controller = new AbortController();
    this.requestControllers.add(controller);
    const timer = setTimeout(() => controller.abort(new Error('Harness request timed out; outcome may already be committed')), timeout);
    try {
      const response = await fetch(`${this.url}/api/harness/v1${path}`, {method, signal: controller.signal,
        headers: {'Authorization': `Bearer ${this.token}`, ...(body === undefined ? {} : {'Content-Type': 'application/json'})},
        ...(body === undefined ? {} : {body: JSON.stringify(body)})});
      let value: unknown;
      const raw = await response.text();
      try { value = JSON.parse(raw); } catch { throw new Error(`Harness returned a non-JSON response (${response.status})`); }
      if (!response.ok) throw new Error(`Harness HTTP ${response.status}: ${print(record(value).detail || record(value).error || value)}`);
      return record(value);
    } finally { clearTimeout(timer); this.requestControllers.delete(controller); }
  }

  /** Canonical Python helper owns service locks/launching; this process owns no Agent. */
  private async ensure(): Promise<void> {
    const code = 'import json, os, signal, threading\ncancel = threading.Event()\nsignal.signal(signal.SIGTERM, lambda *_: cancel.set())\nfrom tsad_harness.bridge import RemoteClient, _runtime_dir\nc = RemoteClient(url=os.environ.get("TSAD_HARNESS_URL") or None)\nc.ensure(cancel=cancel)\nprint(json.dumps({"url": c.url, "tokenFile": str(_runtime_dir() / "access-token")}), flush=True)\n';
    await new Promise<void>((resolvePromise, reject) => {
      let stderr = '', stdout = '', settled = false;
      const child = spawn(this.python, ['-c', code], {cwd: this.root, stdio: ['ignore', 'pipe', 'pipe'], env: {...process.env,
        PYTHONPATH: [join(this.root, 'EasyTSAD/Harness'), process.env.PYTHONPATH].filter(Boolean).join(':'),
        ...(this.explicitUrl ? {TSAD_HARNESS_URL: this.url} : {}), ...(this.explicitToken ? {TSAD_HARNESS_TOKEN: this.token} : {})}});
      this.bootstrap = child;
      const finish = (error?: Error) => { if (settled) return; settled = true; clearTimeout(timer); this.bootstrap = undefined; this.bootstrapCancel = undefined; error ? reject(error) : resolvePromise(); };
      const timer = setTimeout(() => { child.kill('SIGTERM'); finish(new Error('Background startup is still pending; use /connect to retry')); }, 190_000);
      this.bootstrapCancel = () => { child.kill('SIGTERM'); finish(new Error('Terminal client closed; background service keeps running')); };
      child.stdout?.on('data', chunk => { stdout = (stdout + String(chunk)).slice(-8192); });
      child.stderr?.on('data', chunk => { stderr = (stderr + String(chunk)).slice(-3000); });
      child.once('error', error => finish(error));
      child.once('exit', code => {
        if (code === 0) {
          try {
            const metadata = record(JSON.parse(stdout.trim().split('\n').at(-1) || '{}'));
            if (!this.explicitUrl && metadata.url) this.url = text(metadata.url).replace(/\/$/, '');
            if (!this.options.tokenFile && !process.env.TSAD_HARNESS_TOKEN_FILE && metadata.tokenFile) this.tokenFile = text(metadata.tokenFile);
          } catch { finish(new Error('Background helper returned invalid connection metadata')); return; }
        }
        finish(code === 0 ? undefined : new Error(`Background connection failed (${code}): ${stderr}`));
      });
    });
  }

  connect(): Promise<void> {
    if (this.connectPromise) return this.connectPromise;
    if (this.disposed) return Promise.reject(new Error('Terminal client is closed'));
    this.connectPromise = this.open().finally(() => { this.connectPromise = undefined; });
    return this.connectPromise;
  }
  private async open(): Promise<void> {
    const generation = this.generation;
    this.options.onStatus('正在连接后台… 退出不会停止任务');
    await this.discoverConfig();
    await this.readToken();
    if (this.disposed) return;
    let health: Json;
    try { health = await this.api('GET', '/health', undefined, 2500); }
    catch {
      if (this.disposed) return;
      await this.ensure(); await this.readToken();
      health = await this.api('GET', '/health', undefined, 2500);
    }
    if (this.disposed) return;
    if (!health.ok) throw new Error('Invalid Harness health response');
    this.ready = Boolean(health.connected);
    if (!this.ready) this.options.onStatus(`后台已启动，DSH 连接中：${text(health.runtimeError)}`);
    if (this.ready) this.options.onStatus('后台已连接；正在准备会话…');
    if (!this.selectedSessionId && this.ready) {
      if (this.options.resume) {
        let id = this.options.resume;
        if (id === 'latest') {
          const roots = entries((await this.api('GET', '/sessions')).sessions).map(record)
            .filter(row => row.transport !== 'cli' && !row.parentSessionId && !row.parentSession)
            .sort((a, b) => Number(b.updatedAt || 0) - Number(a.updatedAt || 0));
          id = sessionId(roots[0]);
          if (!id) throw new Error('没有可恢复的会话；使用 /new 新建');
        }
        if (this.disposed || generation !== this.generation) return;
        await this.select(id);
      } else await this.createSession(false);
    } else await this.poll();
    if (this.disposed) return;
    this.startBusinessEvents(); this.schedule();
  }

  async select(id: string): Promise<void> {
    if (!id) throw new Error('A session id is required');
    if (this.disposed) return;
    ++this.selectionIntent;
    ++this.generation; this.selectedSessionId = id;
    this.selectedJob = this.options.task || ''; this.job = {};
    this.snapshot = {}; this.history = []; this.notices = []; this.historyRecords.clear(); this.historyHasMore = false; this.historyBefore = undefined; this.streamText = ''; this.cliFinalPending = false;
    this.eventCursor = 0; this.eventEpoch = ''; this.snapshotNeeded = true; this.snapshotDeadline = 0;
    this.commandDescriptors = []; this.commandsNeeded = true; this.published = ''; this.emit();
    this.options.onStatus(`后台已连接 · ${id} · 正在加载历史…`);
    this.loadMetadata(this.generation, id);
    await this.poll();
    this.schedule();
  }

  /** Safe to call in tests and manual refresh; only one network poll is in flight. */
  async poll(): Promise<void> {
    if (this.disposed || this.polling) return;
    this.polling = true;
    const generation = this.generation, id = this.selectedSessionId;
    try {
      if (!this.ready) {
        const health = await this.api('GET', '/health', undefined, 2500);
        if (!this.alive(generation, id)) return;
        this.ready = Boolean(health.connected);
        if (!this.ready) { this.options.onStatus(`后台已启动，DSH 连接中：${text(health.runtimeError)}`); return; }
        if (!id) {
          this.polling = false;
          await this.open(); return;
        }
      }
      if (!id) return;
      this.loadMetadata(generation, id);
      const initialSnapshot = this.snapshotNeeded ? this.api('GET', `/sessions/${pathId(id)}`).then(value => ({value}), error => ({error})) : undefined;
      const live = await this.api('GET', `/sessions/${pathId(id)}/events?after=${this.eventCursor}&epoch=${encodeURIComponent(this.eventEpoch)}`);
      if (!this.alive(generation, id)) return;
      if (live.reset || (this.eventEpoch && text(live.epoch) !== this.eventEpoch)) {
        this.streamText = ''; this.snapshotNeeded = true; this.historyRecords.clear(); this.historyBefore = undefined; this.historyHasMore = false;
      }
      this.eventCursor = Number(live.cursor) || 0; this.eventEpoch = text(live.epoch);
      for (const value of entries(live.events)) {
        const event = record(value), data = record(event.data);
        if (event.sessionId && event.sessionId !== id) continue;
        if (event.type === 'assistant.stream') this.streamText += streamDelta(data);
        else if (event.type === 'text') { this.streamText += text(event.text); this.snapshotNeeded = true; }
        else if (event.type !== 'thinking') this.snapshotNeeded = true;
        if (event.type === 'final') {
          this.streamText = text(event.text) || this.streamText; this.cliFinalPending = true; this.snapshotNeeded = true;
        } else if (['assistant/message', 'turn/end'].includes(text(data.type))) {
          this.streamText = ''; this.snapshotNeeded = true;
        }
      }
      this.emit();
      if (this.snapshotNeeded || ((this.snapshot.running || this.cliFinalPending) && Date.now() >= this.snapshotDeadline)) {
        this.snapshotNeeded = false;
        const loaded = initialSnapshot ? await initialSnapshot : {value: await this.api('GET', `/sessions/${pathId(id)}`)};
        if ('error' in loaded) throw loaded.error;
        const state = loaded.value;
        if (!this.alive(generation, id)) return;
        this.snapshot = state; this.mergeHistory(state);
        this.snapshotDeadline = Date.now() + (state.running ? 4000 : 10_000);
        if (this.cliFinalPending && entries(state.history).some(value => record(value).role === 'assistant')) { this.cliFinalPending = false; this.streamText = ''; }
        if (!state.running && !this.cliFinalPending) this.streamText = '';
        this.selectedJob = text(record(state.binding).jobId || state.jobId) || this.selectedJob;
        // Headless final text is persisted after its final event; keep polling
        // the snapshot even while no more cursor increments arrive.
        if (state.transport === 'cli' || record(state.binding).kind === 'cli' || state.eventEpoch) this.snapshotDeadline = Date.now() + Math.max(500, (this.options.refresh ?? 2) * 1000);
      }
      this.loadMetadata(generation, id);
      this.emit(); this.emitStatus();
    } catch (error) {
      if (this.alive(generation, id)) {
        this.snapshotNeeded = true;
        this.options.onStatus(`会话暂不可用：${errorText(error)}；/new 新建或 /agent 切换，/connect 重连`);
      }
    } finally {
      this.polling = false;
      if (!this.disposed && (generation !== this.generation)) queueMicrotask(() => { void this.poll(); });
    }
  }

  private loadMetadata(generation: number, id: string): void {
    if (this.metadataLoading || !id) return;
    this.metadataLoading = true;
    const jobId = this.selectedJob;
    void Promise.allSettled([
      jobId ? this.api('GET', `/jobs/${pathId(jobId)}`).then(value => {
        if (this.alive(generation, id) && jobId === this.selectedJob) this.job = value;
      }) : Promise.resolve(),
      this.commandsNeeded ? this.api('GET', `/commands?sessionId=${pathId(id)}`).then(value => {
        if (this.alive(generation, id)) { this.commandDescriptors = entries(value.commands).map(record); this.commandsNeeded = false; }
      }) : Promise.resolve(),
    ]).finally(() => {
      this.metadataLoading = false;
      if (this.alive(generation, id)) { this.emit(); }
      else if (!this.disposed) this.loadMetadata(this.generation, this.selectedSessionId);
    });
  }

  async createSession(background = false): Promise<Json> {
    const ticket = randomUUID(), intent = background ? this.selectionIntent : ++this.selectionIntent;
    this.creations.set(ticket, {id: ticket, status: 'starting'});
    this.options.onNotice(`正在${background ? '后台' : ''}创建会话 ${ticket}…`);
    try {
      const result = await this.api('POST', '/sessions', {sessionId: `tsad-${ticket}`});
      const id = sessionId(result);
      this.creations.set(ticket, {id, status: 'created'});
      if (!this.disposed) {
        this.options.onNotice(`会话已创建：${id}；/agent ${id} 切换`);
        if (!background && intent === this.selectionIntent) await this.select(id);
      }
      return result;
    } catch (error) {
      this.creations.set(ticket, {id: ticket, status: 'failed', error: errorText(error)});
      throw error;
    }
  }

  private schedule(): void {
    clearTimeout(this.timer);
    if (!this.disposed) this.timer = setTimeout(() => { void this.poll().finally(() => this.schedule()); }, Math.max(.1, this.options.refresh ?? 2) * 1000);
  }
  private emitStatus(): void {
    const progress = record(this.job.progress), branches = record(this.job.branches);
    const done = Number(progress.completed || 0), total = Number(progress.total || 0);
    const filled = total ? Math.max(0, Math.min(10, Math.round(10 * done / total))) : 0;
    const bar = total ? `[${'█'.repeat(filled)}${'░'.repeat(10 - filled)}] ${done}/${total}` : '';
    const stage = [text(this.job.method), text(this.job.stage), text(this.job.status), bar, text(progress.dataset)].filter(Boolean).join(' ');
    const branchText = Object.entries(branches).map(([role, value]) => `${role}:${text(record(value).status) || 'pending'}`).join(' ');
    this.options.onStatus([`${this.selectedSessionId || '未选择会话'} · ${text(this.snapshot.status) || 'idle'}`, stage, branchText].filter(Boolean).join(' | '));
  }
  private emit(): void {
    if (this.disposed) return;
    let history = this.history;
    if (!this.details) history = history.filter(message => message.role !== 'tool' || message.content.length > 0);
    const cliTranscript = (this.snapshot.transport === 'cli' || this.snapshot.eventEpoch) && entries(this.snapshot.events).some(value => ['text', 'final'].includes(text(record(value).type)));
    const streamAlreadyCommitted = Boolean(cliTranscript) || history.some(message => message.role === 'assistant' && message.content === this.streamText);
    const result = [...history,
      ...(this.streamText && !streamAlreadyCommitted ? [{id: 'remote-live', role: 'assistant', content: this.streamText, streaming: true}] : []), ...this.notices];
    const encoded = JSON.stringify(result);
    if (encoded !== this.published) { this.published = encoded; this.options.onMessages(result); }
  }


  /** Merge raw Session events before projecting so replacements can cross pages. */
  private mergeHistory(state: Json, older = false): void {
    const page = record(state.historyPage), rows = entries(page.records);
    if (!rows.length) { if (!older) this.history = normalizeHistory(state); return; }
    let minimum = Infinity;
    for (const value of rows) {
      const wrapper = record(value), event = wrapper.type === 'event' ? record(wrapper.event) : wrapper;
      const seq = Number(event.seq);
      if (Number.isSafeInteger(seq) && seq >= 0) { this.historyRecords.set(seq, {type: 'event', event}); minimum = Math.min(minimum, seq); }
    }
    if (Number.isFinite(minimum) && (this.historyBefore === undefined || minimum < this.historyBefore)) {
      this.historyBefore = minimum; this.historyHasMore = Boolean(page.hasMore);
    }
    const records = [...this.historyRecords].sort(([a], [b]) => a - b).map(([, value]) => value);
    this.history = normalizeHistory({historyPage: {records}});
  }

  async loadOlder(before?: string): Promise<void> {
    if (!this.selectedSessionId) throw new Error('请先选择会话');
    if (before && before !== 'older' && !/^\d+$/.test(before)) throw new Error('Usage: /history [older|beforeSeq]');
    const cursor = before && before !== 'older' ? Number(before) : this.historyBefore;
    if (cursor === undefined || (!this.historyHasMore && (!before || before === 'older'))) { this.options.onNotice('已加载全部可用历史'); return; }
    const generation = this.generation, id = this.selectedSessionId;
    const state = await this.api('GET', `/sessions/${pathId(id)}?beforeSeq=${cursor}&maxMessages=250`);
    if (!this.alive(generation, id)) return;
    const previous = this.historyBefore;
    this.mergeHistory(state, true); this.emit();
    if (this.historyBefore === previous) { this.historyHasMore = false; this.options.onNotice('没有更早的历史'); }
    else this.options.onNotice(this.historyHasMore ? '已加载更早历史；/history 继续加载' : '已加载全部可用历史');
  }

  async send(value: string, options: {mode?: 'queue' | 'steer'} = {}): Promise<void> {
    const input = value.trim(); if (!input) return;
    if (input.startsWith('/')) { await this.command(input); return; }
    if (!this.selectedSessionId || !this.ready) throw new Error('后台仍在连接；请 /connect 或 /new');
    const id = this.selectedSessionId, generation = this.generation;
    await this.api('POST', `/sessions/${pathId(id)}/messages`, {text: input, mode: options.mode || 'queue', requestId: randomUUID()}, 120_000);
    if (!this.alive(generation, id)) return;
    this.snapshotNeeded = true; await this.poll();
  }

  async command(line: string): Promise<unknown> {
    const input = line.trim(), [name, ...args] = input.slice(1).split(/\s+/);
    if (name === 'history') { await this.loadOlder(args[0]); return; }
    if (name === 'connect') { await this.connect(); return; }
    if (name === 'new') {
      if (args.length && (args.length !== 1 || args[0] !== '--background')) throw new Error('Usage: /new [--background]');
      return this.createSession(args[0] === '--background');
    }
    if (name === 'sessions') {
      const result = await this.api('GET', '/sessions');
      const rows = entries(result.sessions).map(record);
      this.options.onNotice([...rows.map(row => `${sessionId(row)} · ${text(row.status) || 'inactive'}${sessionId(row) === this.selectedSessionId ? ' · 当前' : ''}`),
        ...[...this.creations.values()].filter(row => row.status !== 'created').map(row => `${row.id} · ${row.status}${row.error ? ` · ${row.error}` : ''}`),
        '/new [--background] 新建；/agent <id> 切换'].join('\n'));
      return result;
    }
    if (name === 'resume') {
      if (args.length !== 1) throw new Error('Usage: /resume <sessionId>');
      await this.select(args[0]); return;
    }
    if (name === 'agent' && args.length) { await this.select(args[0]); return; }
    const generation = this.generation, id = this.selectedSessionId;
    // GPU preferences are explicit structured domain arguments, never silently
    // appended to natural-language paper input or native command grammar.
    let result: Json;
    if (this.options.gpus && ['run', 'resume', 'retry'].includes(name) && (args[0] || this.selectedJob)) {
      const target = args[0] || this.selectedJob;
      const batches = entries((await this.api('GET', '/batches')).batches).map(record);
      const family = batches.some(batch => batch.id === target) ? 'batches' : 'jobs';
      result = await this.api('POST', `/${family}/${pathId(target)}/${name}`, {gpus: this.options.gpus, requestId: randomUUID()}, 120_000);
    } else {
      const enriched = this.selectedJob && ['run', 'pause', 'resume', 'stop', 'retry', 'logs', 'scores', 'report'].includes(name) && !args.length ? `${input} ${this.selectedJob}` : input;
      result = await this.api('POST', '/commands', {sessionId: id, line: enriched, requestId: randomUUID(), ...(this.options.gpus ? {gpus: this.options.gpus} : {})}, 120_000);
    }
    if (!this.alive(generation, id)) return result;
    if (result.selected && sessionId(result)) { await this.select(sessionId(result)); return result; }
    const jobs = entries(result.jobs);
    if (jobs.length && ['reproduce', 'run', 'resume', 'retry'].includes(name)) this.selectedJob = typeof jobs[0] === 'string' ? jobs[0] : sessionId(jobs[0]);
    else if (result.stage && result.id) this.selectedJob = text(result.id);
    if (name === 'help') result = {...result, clientCommands: {new: '新建并切换；/new --background 后台创建', sessions: '列出后台会话及创建状态', resume: '恢复指定会话：/resume <sessionId>', agent: '切换会话：/agent <sessionId>', connect: '重连后台', details: '展开或折叠工具输出', history: '加载更早历史：/history [older]'}};
    this.notices.push({id: `command-${randomUUID()}`, role: 'system', content: `${input}\n\n\`\`\`json\n${print(result)}\n\`\`\``});
    this.options.onNotice(`完成 ${input}`);
    this.snapshotNeeded = true; this.commandsNeeded = true; this.emit(); await this.poll();
    return result;
  }

  async cancel(): Promise<void> {
    if (!this.selectedSessionId) return;
    const id = this.selectedSessionId, generation = this.generation;
    await this.api('POST', `/sessions/${pathId(id)}/cancel`, {});
    if (this.alive(generation, id)) { this.snapshotNeeded = true; await this.poll(); }
  }

  /** One business stream per terminal; selected-session tokens remain cursor-polled. */
  private startBusinessEvents(): void {
    this.streamController?.abort();
    const controller = new AbortController(); this.streamController = controller;
    void (async () => {
      while (!controller.signal.aborted && !this.disposed) {
        try {
          const response = await fetch(`${this.url}/api/harness/v1/events`, {headers: {Authorization: `Bearer ${this.token}`, ...(this.businessCursor ? {'Last-Event-ID': this.businessCursor} : {})}, signal: controller.signal});
          if (!response.ok || !response.body || !response.headers.get('content-type')?.includes('text/event-stream')) throw new Error(`Business events HTTP ${response.status}`);
          const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
          for (;;) {
            const value = await reader.read(); if (value.done) break;
            buffer += decoder.decode(value.value, {stream: true}).replace(/\r\n/g, '\n');
            let end: number;
            while ((end = buffer.indexOf('\n\n')) >= 0) {
              const packet = buffer.slice(0, end); buffer = buffer.slice(end + 2);
              for (const field of packet.split('\n')) if (field.startsWith('id:')) this.businessCursor = field.slice(3).trim();
              if (packet.split('\n').some(field => field.startsWith('data:'))) { this.snapshotNeeded = true; void this.poll(); }
            }
          }
        } catch (error) { if (!controller.signal.aborted && !this.disposed) this.options.onStatus(`业务事件重连中：${errorText(error)}`); }
        if (!controller.signal.aborted) await new Promise<void>(resolvePromise => {
          const finish = () => { clearTimeout(timer); controller.signal.removeEventListener('abort', finish); resolvePromise(); };
          const timer = setTimeout(finish, 1000); controller.signal.addEventListener('abort', finish, {once: true});
        });
      }
    })();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true; this.ready = false; ++this.generation;
    clearTimeout(this.timer); this.streamController?.abort();
    for (const controller of this.requestControllers) controller.abort();
    this.bootstrapCancel?.(); this.bootstrap?.kill('SIGTERM');
    // Only the short-lived canonical helper belongs to this UI. Never signal
    // the detached service, DSH host, worker, evaluation or process group.
  }
}
