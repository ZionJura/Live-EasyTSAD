/** Small durable task board for the pinned rc1 runtime, which has no Team package. */
import {readFile, writeFile, mkdir, rename} from 'node:fs/promises';
import {join} from 'node:path';
import {randomUUID} from 'node:crypto';
import {HttpError} from './support.mjs';

const copy = value => structuredClone(value);
const fail = (message, status = 400) => { throw new HttpError(status, message); };
function identifier(value, label = 'id') {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(value)) fail(`Invalid Team ${label}`);
  return value;
}
function words(value, label, max, empty = false) {
  if (typeof value !== 'string' || value.length > max || (!empty && !value.trim())) fail(`Invalid Team ${label}`);
  return value.trim();
}
function strings(value, label, limit = 32) {
  if (!Array.isArray(value) || value.length > limit || value.some(item => typeof item !== 'string' || !item.trim() || item.length > 1000)) fail(`Invalid Team ${label}`);
  if (new Set(value).size !== value.length) fail(`Duplicate Team ${label}`);
  return [...value];
}
function fields(request, allowed) {
  for (const key of Object.keys(request)) if (!allowed.includes(key)) fail(`Unsupported Team field: ${key}`);
}
function validate(board) {
  if (!board || typeof board !== 'object' || typeof board.enabled !== 'boolean' || !Number.isSafeInteger(board.nextId) || board.nextId < 1 || !Array.isArray(board.tasks) || board.tasks.length > 256) fail('Invalid persisted Team board');
  const tasks = new Map();
  for (const task of board.tasks) {
    identifier(task.id, 'task id');
    if (/^\d+$/.test(task.id) && Number(task.id) >= board.nextId) fail('Invalid Team task sequence');
    if (tasks.has(task.id) || !Number.isSafeInteger(task.revision) || task.revision < 1 || !['pending', 'in_progress', 'completed', 'deleted'].includes(task.status)) fail('Invalid Team task state');
    words(task.subject, 'subject', 500); words(task.description, 'description', 20000, true);
    strings(task.blockedBy, 'dependencies'); strings(task.writeScopes, 'write scopes');
    if (task.ownerId !== undefined) identifier(task.ownerId, 'owner');
    tasks.set(task.id, task);
  }
  const visiting = new Set(), done = new Set();
  function visit(id) {
    if (visiting.has(id)) fail('Team dependencies must not contain a cycle');
    if (done.has(id)) return;
    visiting.add(id);
    for (const dependency of tasks.get(id).blockedBy) {
      identifier(dependency, 'dependency');
      if (!tasks.has(dependency) || tasks.get(dependency).status === 'deleted') fail('Team dependency does not exist');
      visit(dependency);
    }
    visiting.delete(id); done.add(id);
  }
  for (const task of tasks.values()) if (task.status !== 'deleted') visit(task.id);
}

export class TeamAdapter {
  static async open(home, rootOf = caller => caller.id) {
    const adapter = new TeamAdapter(home, rootOf);
    try {
      const state = JSON.parse(await readFile(adapter.path, 'utf8'));
      if (state.version !== 1 || !state.boards || typeof state.boards !== 'object' || Array.isArray(state.boards)) fail('Invalid persisted Team state');
      for (const [id, board] of Object.entries(state.boards)) { identifier(id, 'root session'); validate(board); }
      adapter.state = state;
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    return adapter;
  }
  constructor(home, rootOf) {
    this.home = home; this.path = join(home, 'tsad-team-boards.json'); this.rootOf = rootOf;
    this.state = {version: 1, boards: {}}; this.pending = Promise.resolve();
  }
  root(caller) { identifier(caller?.id, 'caller'); return identifier(this.rootOf(caller), 'root session'); }
  board(state, root) { return Object.hasOwn(state.boards, root) ? state.boards[root] : (state.boards[root] = {enabled: true, nextId: 1, tasks: []}); }
  snapshot(caller) {
    const root = this.root(caller), board = Object.hasOwn(this.state.boards, root) ? this.state.boards[root] : {enabled: true, tasks: []};
    return {enabled: board.enabled, tasks: this.views(board), implementation: 'tsad-adapter'};
  }
  views(board) {
    return copy(board.tasks.filter(task => task.status !== 'deleted').map(task => ({...task,
      ready: task.status === 'pending' && task.blockedBy.every(id => board.tasks.find(other => other.id === id)?.status === 'completed'),
      writeScopeWarnings: []})));
  }
  listTasks(caller) { return this.snapshot(caller).tasks; }
  async mutate(caller, operation, signal) {
    const root = this.root(caller);
    const pending = this.pending.then(async () => {
      signal?.throwIfAborted();
      const state = copy(this.state), board = this.board(state, root);
      const result = operation(board, root);
      validate(board);
      await mkdir(this.home, {recursive: true, mode: 0o700});
      const temp = `${this.path}.${randomUUID()}.tmp`;
      await writeFile(temp, JSON.stringify(state, null, 2) + '\n', {mode: 0o600});
      await rename(temp, this.path);
      this.state = state;
      return copy(result);
    });
    this.pending = pending.catch(() => {});
    return pending;
  }
  async setEnabled(caller, enabled, signal) {
    if (caller.id !== this.root(caller)) fail('Change Team mode from the root session', 403);
    if (typeof enabled !== 'boolean') fail('Team enabled must be boolean');
    return this.mutate(caller, board => { board.enabled = enabled; return {enabled}; }, signal);
  }
  async createTask(caller, request, signal) {
    if (!request || typeof request !== 'object' || Array.isArray(request)) fail('Team task is required');
    fields(request, ['subject', 'description', 'blockedBy', 'writeScopes']);
    return this.mutate(caller, board => {
      if (!board.enabled) fail('Team board is disabled', 409);
      if (board.tasks.length >= 256) fail('Team task limit (256) reached', 409);
      const task = {id: String(board.nextId++), revision: 1,
        subject: words(request.subject, 'subject', 500), description: words(request.description ?? '', 'description', 20000, true),
        status: 'pending', blockedBy: strings(request.blockedBy ?? [], 'dependencies'), writeScopes: strings(request.writeScopes ?? [], 'write scopes')};
      board.tasks.push(task);
      return this.views(board).find(item => item.id === task.id);
    }, signal);
  }
  async updateTask(caller, request, signal) {
    if (!request || typeof request !== 'object' || Array.isArray(request)) fail('Team update is required');
    fields(request, ['taskId', 'id', 'expectedRevision', 'action', 'subject', 'description', 'blockedBy', 'writeScopes', 'owner']);
    const id = identifier(request.taskId ?? request.id, 'task id');
    return this.mutate(caller, (board, root) => {
      if (!board.enabled) fail('Team board is disabled', 409);
      const task = board.tasks.find(item => item.id === id);
      if (!task || task.status === 'deleted') fail('Team task does not exist', 404);
      if (!Number.isSafeInteger(request.expectedRevision) || request.expectedRevision !== task.revision) fail('Team task revision changed; read the board before updating', 409);
      if (task.ownerId && task.ownerId !== caller.id && caller.id !== root) fail('Team task belongs to another agent', 403);
      const ready = task.blockedBy.every(dependency => board.tasks.find(item => item.id === dependency)?.status === 'completed');
      switch (request.action) {
        case 'claim':
          if (task.status !== 'pending' || !ready || (task.ownerId && task.ownerId !== caller.id)) fail('Team task is not ready to claim', 409);
          task.ownerId = caller.id; task.status = 'in_progress'; break;
        case 'release':
          if (task.status !== 'in_progress') fail('Only an active Team task can be released', 409);
          delete task.ownerId; task.status = 'pending'; break;
        case 'complete':
          if (!['pending', 'in_progress'].includes(task.status) || !ready) fail('Team task cannot complete while blocked or closed', 409);
          task.status = 'completed'; break;
        case 'reopen':
          if (task.status !== 'completed') fail('Only a completed Team task can reopen', 409);
          task.status = 'pending'; delete task.ownerId; break;
        case 'delete': task.status = 'deleted'; break;
        case 'reassign':
          if (caller.id !== root) fail('Only the Team root can reassign tasks', 403);
          if (request.owner === undefined) delete task.ownerId;
          else task.ownerId = identifier(request.owner, 'owner');
          break;
        case 'set_dependencies':
          if (task.status !== 'pending') fail('Only a pending Team task can change dependencies', 409);
          task.blockedBy = strings(request.blockedBy, 'dependencies'); break;
        case 'edit':
          if (request.subject === undefined && request.description === undefined && request.writeScopes === undefined) fail('Team edit requires changed fields');
          if (request.subject !== undefined) task.subject = words(request.subject, 'subject', 500);
          if (request.description !== undefined) task.description = words(request.description, 'description', 20000, true);
          if (request.writeScopes !== undefined) task.writeScopes = strings(request.writeScopes, 'write scopes');
          break;
        default: fail('Unknown Team task action');
      }
      task.revision++;
      return this.views(board).find(item => item.id === task.id) || {...task, ready: false, writeScopeWarnings: []};
    }, signal);
  }
}
