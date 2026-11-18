import { timingSafeEqual } from 'node:crypto';

export const VERSION = '0.2.2-rc.1';
export const ROLES = ['lead', 'researcher', 'implementer', 'supervisor', 'reporter'];
export const READ_TOOLS = new Set(['tasks', 'scores', 'search', 'fetch', 'status', 'diagnose']);
export const DOMAIN_TOOLS = ['reproduce', 'tasks', 'run', 'pause', 'resume', 'stop', 'retry', 'correct', 'scores', 'search', 'fetch', 'status', 'diagnose', 'complete', 'environment'];
export const READ_NATIVE = new Set(['read', 'read_image', 'glob', 'grep', 'web_search', 'web_fetch', 'load_skill', 'skill', 'todo_write', 'get_goal', 'list_agents']);

export class HttpError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export function authorized(header, token) {
  if (!token || typeof header !== 'string') return false;
  const expected = Buffer.from(`Bearer ${token}`), actual = Buffer.from(header);
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}

export function text(value, name = 'text', max = 200_000) {
  if (typeof value !== 'string' || !value.trim() || value.length > max) throw new HttpError(400, `${name} must be non-empty text, at most ${max} characters`);
  return value;
}

export function roleGuard(binding, tool, args = {}) {
  if (tool === 'tsad_team' && binding?.role !== 'lead') return 'The TSAD Team task board is available to the lead role only.';
  if (!binding) return;
  if (binding.jobId && /(?:update_goal|complete_goal|goal_update)/.test(tool)
      && (args.status === 'complete' || args.action === 'complete' || args.phase === 'complete')) {
    return 'The TSAD scheduler owns job completion; report evidence and await domain verification.';
  }
  if (['supervisor', 'researcher', 'reporter'].includes(binding.role)) {
    if (tool.startsWith('tsad_') ? !READ_TOOLS.has(tool.slice(5)) : !READ_NATIVE.has(tool)) {
      return `${binding.role} is read-only; shell, file mutation, delegation and workflow control are unavailable.`;
    }
  }
  if (binding.role === 'lead' && ['bash', 'pwsh', 'write', 'edit', 'apply_patch', 'run_code', 'workflow', 'plugin_manager'].includes(tool)) {
    return 'The coordinator uses domain tools and delegates method changes; direct shell/file mutation is unavailable.';
  }
  if (binding.role === 'implementer' && ['bash', 'pwsh', 'run_code', 'workflow', 'plugin_manager', 'subagent', 'subagent_fork'].includes(tool)) {
    return 'The implementer uses leased file tools and tsad_environment/tsad_run; unrestricted shell and delegation are unavailable.';
  }
  if (binding.role === 'implementer' && !binding.jobId && ['write', 'edit', 'apply_patch'].includes(tool)) return 'Method changes require a bound TSAD job and write lease.';
}

export class EventJournal {
  constructor(limit = 5000) { this.limit = limit; this.cursor = 0; this.events = []; }
  append(sessionId, type, data) {
    const event = {seq: ++this.cursor, sessionId, type, data};
    this.events.push(event);
    if (this.events.length > this.limit) this.events.shift();
    return event;
  }
  read(sessionId, after = 0) {
    return {events: this.events.filter(e => e.seq > after && e.sessionId === sessionId), cursor: this.cursor,
      reset: after > this.cursor || (after > 0 && this.events.length > 0 && after < this.events[0].seq - 1)};
  }
}

export class Semaphore {
  constructor(limit) { this.limit = limit; this.active = 0; this.waiters = []; }
  async acquire(signal) {
    signal?.throwIfAborted();
    if (this.active >= this.limit) {
      await new Promise((resolve, reject) => {
        const waiter = {resolve, reject, signal};
        waiter.abort = () => { this.waiters = this.waiters.filter(w => w !== waiter); reject(signal.reason); };
        signal?.addEventListener('abort', waiter.abort, {once: true});
        this.waiters.push(waiter);
      });
    } else this.active++;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const waiter = this.waiters.shift();
      if (waiter) { waiter.signal?.removeEventListener('abort', waiter.abort); waiter.resolve(); }
      else this.active--;
    };
  }
}

export async function readBody(req, limit = 1_000_000) {
  let size = 0; const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new HttpError(413, 'Request body exceeds 1 MB');
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  let value;
  try { value = JSON.parse(Buffer.concat(chunks).toString('utf8')); }
  catch { throw new HttpError(400, 'Expected JSON body'); }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new HttpError(400, 'Expected JSON object');
  return value;
}
