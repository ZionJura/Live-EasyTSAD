/** Native Cordis plugin. No SDK subprocess or simulated Agent loop. */
import { createServer } from 'node:http';
import { acquirePermit } from './model-permits.mjs';
import { sessionHistory } from './history.mjs';
import { TeamAdapter } from './team-adapter.mjs';
import { randomUUID } from 'node:crypto';
import { readFile, writeFile, mkdir, rename, realpath } from 'node:fs/promises';
import { join, resolve, dirname, basename, sep } from 'node:path';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { VERSION, ROLES, DOMAIN_TOOLS, READ_TOOLS, EventJournal, Semaphore, HttpError, authorized, readBody, roleGuard, text } from './support.mjs';

export const name = 'tsad-native-bridge';
export const inject = ['agents', 'sessions', 'sessionPersistence', 'sessionController', 'subagents', 'commands', 'tools', 'goals', 'web', 'sessionProjections'];

export async function apply(ctx) {
  const env = process.env;
  const token = env.TSAD_BRIDGE_TOKEN;
  if (!token || token.length < 16) throw new Error('TSAD_BRIDGE_TOKEN must contain at least 16 characters');
  const control = new URL(env.TSAD_CONTROL_URL || 'http://127.0.0.1:8767');
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(control.hostname)) throw new Error('TSAD_CONTROL_URL must be loopback');
  const home = text(env.DSH_HOME, 'DSH_HOME');
  const bindingsPath = join(home, 'tsad-role-bindings.json');
  const bindings = new Map();
  try { for (const [id, binding] of Object.entries(JSON.parse(await readFile(bindingsPath, 'utf8')))) bindings.set(id, binding); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  let saveChain = Promise.resolve();
  const save = () => {
    const body = JSON.stringify(Object.fromEntries(bindings), null, 2);
    const operation = saveChain.then(async () => { await mkdir(home, {recursive: true}); await writeFile(`${bindingsPath}.tmp`, body, {mode: 0o600}); await rename(`${bindingsPath}.tmp`, bindingsPath); });
    saveChain = operation.catch(error => ctx.logger.error(`TSAD binding persistence failed: ${error.message}`));
    return operation;
  };
  const epoch = randomUUID(), journal = new EventJournal(), errors = new Map(), steps = new Map(), timers = new Map();
  const teamRoot = caller => {
    let id = caller.id;
    const seen = new Set();
    while (!seen.has(id)) {
      seen.add(id);
      const parent = bindings.get(id)?.parentSessionId || ctx.sessions.get(id)?.header.parentSession;
      if (!parent) return id;
      id = parent;
    }
    throw new HttpError(409, 'Invalid Team session ancestry');
  };
  const teams = await TeamAdapter.open(home, teamRoot);
  const maxSteps = Number(env.TSAD_MAX_STEPS || 120), maxWallMs = Number(env.TSAD_MAX_WALL_SECONDS || 5400) * 1000;
  const capacity = new Semaphore(Number(env.TSAD_MAX_MODEL_CONCURRENCY || 4));
  const stop = new AbortController();
  let ready = false;
  const signal = () => stop.signal;
  const domain = async (tool, args, agentOrId, callSignal = signal()) => {
    const id = typeof agentOrId === 'string' ? agentOrId : agentOrId?.id;
    const binding = bindings.get(id);
    const denial = roleGuard(binding, `tsad_${tool}`, args);
    if (denial) throw new HttpError(403, denial);
    const response = await fetch(new URL(`/internal/tools/${tool}`, control), {
      method: 'POST', headers: {'content-type': 'application/json', authorization: `Bearer ${env.TSAD_CONTROL_TOKEN || token}`},
      body: JSON.stringify({arguments: args, sessionId: id, role: binding?.role || 'lead', jobId: binding?.jobId}),
      signal: AbortSignal.any([callSignal, AbortSignal.timeout(120_000)]),
    });
    const result = await response.json();
    if (!response.ok) throw new HttpError(response.status, result.error?.message || result.error || `TSAD ${tool} failed`);
    return result;
  };
  const teamOperation = async (current, args, callSignal = signal()) => {
    const denial = roleGuard(bindings.get(current.id), 'tsad_team', args);
    if (denial) throw new HttpError(403, denial);
    let result;
    if (args.action === 'list') return teams.snapshot(current);
    if (args.action === 'enable') result = await teams.setEnabled(current, args.enabled ?? true, callSignal);
    else if (args.action === 'create') result = await teams.createTask(current, args.task, callSignal);
    else if (args.action === 'update') result = await teams.updateTask(current, args.task, callSignal);
    else throw new HttpError(400, 'Team action must be list, enable, create or update');
    journal.append(current.id, 'team.updated', {team: teams.snapshot(current)});
    return result;
  };

  ctx.on('session/event', (session, event) => {
    journal.append(session.id, 'session.event', event);
    if (event.type === 'step/start') steps.set(session.id, (steps.get(session.id) || 0) + 1);
    if (event.type === 'turn/end' && event.data?.reason) {
      const reason = event.data.reason;
      const kind = typeof reason === 'string' ? reason : reason.kind;
      if (!['completed', 'stop'].includes(kind)) errors.set(session.id, event.data);
    }
  });
  ctx.on('agent/status', ({agent, status}) => {
    clearTimeout(timers.get(agent.id)); timers.delete(agent.id);
    if (status === 'running') {
      const timer = setTimeout(() => {
        errors.set(agent.id, {message: 'TSAD model wall-clock budget exhausted'}); agent.cancel('user');
      }, maxWallMs);
      timer.unref(); timers.set(agent.id, timer);
    }
    journal.append(agent.id, 'session.status', {status, running: status === 'running'});
  });
  ctx.on('agent/error', ({agent, error}) => { errors.set(agent.id, {message: error.message}); journal.append(agent.id, 'session.error', {message: error.message}); });
  ctx.on('agent/assistant-stream', payload => {
    const id = payload.agent?.id || payload.sessionId;
    if (id) { const {agent, ...data} = payload; journal.append(id, 'assistant.stream', data); }
  });
  // Every real provider stream, including compaction and title requests, shares
  // the cap. The permit lives through iteration and is released on abort/error.
  ctx.on('llm/stream', async function* (options, next) {
    const release = await capacity.acquire(options.signal);
    let releaseShared;
    try {
      releaseShared = await acquirePermit(env.TSAD_MODEL_PERMITS_DIR || join(home, '../model-permits'), capacity.limit, options.signal);
      yield* next();
    } finally { await releaseShared?.(); release(); }
  });
  ctx.on('agent/created', async ({agent}) => {
    if (!bindings.has(agent.id)) {
      const parentSessionId = agent.session.header.parentSession;
      const parent = bindings.get(parentSessionId);
      if (parent) { bindings.set(agent.id, {...parent, parentSessionId, inherited: true}); await save(); }
    }
  });
  ctx.on('agent/pre-step', async ({agent}, next) => {
    const binding = bindings.get(agent.id);
    if (binding && (steps.get(agent.id) || 0) >= (binding.maxSteps || maxSteps)) {
      errors.set(agent.id, {message: 'TSAD model step budget exhausted'});
      agent.cancel('user');
    }
    return next();
  });
  ctx.effect(() => ctx.tools.guard(exec => roleGuard(bindings.get(exec.agent?.id), exec.name, exec.arguments)));
  ctx.on('tools/pre-execute', async (exec, next) => {
    const binding = bindings.get(exec.agent?.id);
    const denied = roleGuard(binding, exec.name, exec.arguments);
    if (denied) return {kind: 'deny', reason: denied};
    if (binding?.jobId && binding.role === 'implementer' && ['write', 'edit', 'apply_patch'].includes(exec.name)) {
      const result = await domain('diagnose', {jobId: binding.jobId, operation: 'check_write_lease', tool: exec.name, arguments: exec.arguments}, exec.agent, exec.signal);
      if (result.leaseGranted !== true) return {kind: 'deny', reason: 'The TSAD method write lease is not held by this job.'};
      const file = exec.arguments?.file_path || exec.arguments?.path;
      if (typeof file !== 'string') return {kind: 'deny', reason: 'A concrete file path is required for leased writes.'};
      const roots = [result.allowedMethodDir, result.allowedEnvironmentDir, result.allowedTaskDir].filter(Boolean);
      async function canonical(path) {
        try { return await realpath(path); }
        catch (error) { if (error.code !== 'ENOENT') throw error; const parent = dirname(path); if (parent === path) throw error; return join(await canonical(parent), basename(path)); }
      }
      const target = await canonical(resolve(exec.agent.session.header.cwd || process.cwd(), file));
      const allowed = await Promise.all(roots.map(root => canonical(resolve(root))));
      if (!allowed.some(root => target === root || target.startsWith(root + sep))) return {kind: 'deny', reason: 'This path is outside the assigned method, environment and task directories.'};
    }
    return next();
  });

  for (const tool of DOMAIN_TOOLS) {
    ctx.effect(() => ctx.tools.register(defineTool({
      name: `tsad_${tool}`, description: `TSAD ${tool}: use the durable experiment controller. Accept structured fields in arguments; do not launch training through shell.`,
      parameters: {arguments: {type: 'json', required: true, description: 'Structured TSAD operation arguments, including method/jobId, papers, datasets, or correction text as appropriate.'}},
      output: {schema: {type: 'json'}, render: (_args, value) => [{type: 'text', text: JSON.stringify(value)}]},
      execute: ({arguments: args}, exec) => domain(tool, args, exec.agent, exec.signal),
    }), {when: ({agent}) => !roleGuard(bindings.get(agent?.id), `tsad_${tool}`)}));
  }
  ctx.effect(() => ctx.tools.register(defineTool({
    name: 'tsad_team', description: 'Maintain the persistent TSAD Team task board. action=list reads it; action=enable takes enabled; action=create takes task={subject,description,blockedBy?,writeScopes?}; action=update takes task={taskId,expectedRevision,action,...}. Update actions: claim,release,edit,set_dependencies,complete,reopen,reassign,delete. Read the latest revision before updating. Advisory tasks do not control experiment stages.',
    parameters: {arguments: {type: 'json', required: true, description: 'Team action and optional task/enable fields.'}},
    output: {schema: {type: 'json'}, render: (_args, value) => [{type: 'text', text: JSON.stringify(value)}]},
    execute: ({arguments: args}, exec) => teamOperation(exec.agent, args, exec.signal),
  }), {when: ({agent}) => !roleGuard(bindings.get(agent?.id), 'tsad_team')}));
  const slash = ['reproduce', 'tasks', 'run', 'pause', 'resume', 'stop', 'retry', 'correct', 'scores', 'search', 'fetch', 'status', 'diagnose', 'environment'];
  for (const command of slash) {
    // The registry is authoritative for both the native Web and thin TUI.
    ctx.effect(() => ctx.commands.register({name: command, description: `TSAD ${command}`, input: {hint: 'Arguments or natural-language instruction'},
      handler: async ({agent, rawInput, signal: callSignal}) => {
        try { return {kind: 'success', text: JSON.stringify(await domain(command, {input: rawInput.trim()}, agent, callSignal))}; }
        catch (error) { return {kind: 'error', text: error.message}; }
      },
    }));
  }
  {
    const endpoint = env.TSAD_SEARCH_URL ? new URL(env.TSAD_SEARCH_URL) : null;
    ctx.effect(() => ctx.web.registerSearchProvider({id: 'tsad-searxng', title: 'SearXNG', description: {en: 'Configured TSAD research search endpoint'},
      available: () => true, ready: async () => endpoint !== null,
      async search(request, callSignal) {
        if (!endpoint) throw new Error('Configure TSAD_SEARCH_URL to enable web search');
        const url = new URL(endpoint); url.searchParams.set('q', request.query); url.searchParams.set('format', 'json');
        const response = await fetch(url, {signal: AbortSignal.any([callSignal, AbortSignal.timeout(30_000)])});
        if (!response.ok) throw new Error(`SearXNG search returned ${response.status}`);
        const result = await response.json(), limit = request.maxResults || 10;
        if (!Array.isArray(result.results)) throw new Error('SearXNG response is missing results');
        return {sources: result.results.slice(0, limit).map(row => ({url: row.url, title: row.title, snippet: row.content})), truncated: result.results.length > limit};
      },
    }));
  }

  async function address(id) {
    // Quiescent native children are removed from the live Session registry.
    // Their persisted header, not a TSAD role binding, owns direct ancestry.
    const header = ctx.sessions.get(id)?.header || (await ctx.sessionPersistence.stat(id, {signal: signal()}))?.header;
    if (header?.origin === 'subagent') return {kind: 'subagent', parentSessionId: header.parentSession, childSessionId: id, mode: 'unknown'};
    const binding = bindings.get(id);
    if (binding?.parentSessionId) return {kind: 'subagent', parentSessionId: binding.parentSessionId, childSessionId: id, mode: 'continuable'};
    return {kind: 'session', sessionId: id};
  }
  async function agent(id) {
    const existing = ctx.agents.get(id);
    if (existing) return existing;
    const target = await address(id);
    if (target.kind === 'subagent') throw new HttpError(409, `Child Agent ${id} is inactive. Send it a real message through its direct parent ${target.parentSessionId} before using native controls or resuming its children. The native runtime has no restore-only subagent API.`);
    const result = await ctx.sessionController.resolveAgent(id);
    if (result.error) throw new HttpError(404, result.error.message);
    return result.agent;
  }
  async function snapshot(id, query) {
    const target = await address(id);
    const maxMessages = Number(query.get('maxMessages') ?? 250);
    const beforeSeq = query.has('beforeSeq') ? Number(query.get('beforeSeq')) : undefined;
    if (!Number.isSafeInteger(maxMessages) || maxMessages < 1 || (beforeSeq !== undefined && (!Number.isSafeInteger(beforeSeq) || beforeSeq < 0))) {
      throw new HttpError(400, 'maxMessages must be a positive integer and beforeSeq a non-negative integer');
    }
    const page = await ctx.sessionController.events({address: target, maxMessages, ...(beforeSeq === undefined ? {} : {beforeSeq})}, signal());
    const projection = await ctx.sessionController.projections({sessionId: id}, signal());
    const values = projection?.values || {};
    const current = ctx.agents.get(id);
    return {sessionId: id, running: current?.status === 'running', status: current?.status || 'inactive', lastError: errors.get(id) || null,
      history: sessionHistory(page), historyPage: page, projections: {...values, agentTeam: teams.snapshot({id})},
      goal: values.goal || null, todos: values.todos || [], team: teams.snapshot({id}), binding: bindings.get(id) || null};
  }
  async function route(method, url, body) {
    const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
    if (method === 'GET' && url.pathname === '/health') return {ok: true, ready, version: VERSION, epoch, modelConcurrency: capacity.limit, teamImplementation: 'tsad-adapter',
      capabilities: {nativeSession: true, nativeCommands: true, subagents: true, team: true, goal: true, skills: true, fetch: true, search: Boolean(env.TSAD_SEARCH_URL)}};
    if (method === 'GET' && url.pathname === '/sessions') return {sessions: (await ctx.sessionController.list({}, signal())).items};
    if (method === 'POST' && url.pathname === '/sessions') {
      const id = body.sessionId || `tsad-${randomUUID()}`;
      bindings.set(id, {role: 'lead', maxSteps: Number(body.maxSteps || maxSteps), systemPrompt: body.systemPrompt}); await save();
      const result = await ctx.sessionController.create({sessionId: id, cwd: resolve(body.cwd || env.TSAD_ROOT || process.cwd()), agentPreset: 'tsad-lead'});
      if (body.systemPrompt) {
        const current = await agent(result.sessionId);
        current.inject({id: randomUUID(), role: 'user', source: {kind: 'user'}, content: [{type: 'text', text: body.systemPrompt}]});
      }
      // Native JSONL persistence defers a blank session header until append or
      // explicit flush. A created conversation must survive a Host restart.
      await ctx.sessions.flush(ctx.sessions.get(result.sessionId));
      return result;
    }
    if (parts[0] === 'sessions' && parts[1]) {
      const id = parts[1];
      if (method === 'GET' && parts.length === 2) return snapshot(id, url.searchParams);
      if (method === 'GET' && parts[2] === 'events') return {...journal.read(id, Number(url.searchParams.get('after') || 0)), epoch,
        reset: (url.searchParams.has('epoch') && url.searchParams.get('epoch') !== epoch) || journal.read(id, Number(url.searchParams.get('after') || 0)).reset};
      if (method === 'POST' && parts[2] === 'messages') {
        text(body.text); const requestId = body.requestId || randomUUID(), mode = body.mode || 'queue';
        if (!['queue', 'steer'].includes(mode)) throw new HttpError(400, 'mode must be queue or steer');
        const target = await address(id); errors.delete(id); steps.set(id, 0);
        // Native delivery requires the exact live direct parent authority.
        // Resolving an ordinary parent restores its controls without waking it.
        if (target.kind === 'subagent') await agent(target.parentSessionId);
        const result = target.kind === 'subagent'
          ? await ctx.subagents.prompt({parentSessionId: target.parentSessionId, childSessionId: id, mode: 'continuable', requestId, delivery: mode, content: [{type: 'text', text: body.text}]}, signal())
          : await ctx.sessionController.prompt({sessionId: id, requestId, mode, content: [{type: 'text', text: body.text}]}, signal());
        return {sessionId: id, requestId, ...result};
      }
      if (method === 'POST' && parts[2] === 'cancel') {
        const target = await address(id);
        if (target.kind === 'subagent') return ctx.subagents.interruptByParent(id, target.parentSessionId, 'continuable');
        return ctx.sessionController.cancel({sessionId: id});
      }
      if (method === 'POST' && parts[2] === 'goals') {
        const current = await agent(id), goal = ctx.goals.get(current);
        if (body.action === 'set') return ctx.goals.create(current, {objective: text(body.objective, 'objective'), maxGoalRounds: body.maxGoalRounds || 24});
        if (!goal) throw new HttpError(409, 'No current goal');
        if (body.action === 'disarm') return ctx.goals.disarm(current);
        if (!['pause', 'resume', 'complete'].includes(body.action)) throw new HttpError(400, 'Unknown goal action');
        return ctx.goals[body.action](current, {id: goal.id, revision: goal.revision});
      }
      if (method === 'POST' && parts[2] === 'team') {
        return teamOperation(await agent(id), body);
      }
    }
    if (method === 'GET' && url.pathname === '/agents') {
      const id = text(url.searchParams.get('sessionId'), 'sessionId');
      return {agents: [{sessionId: id, id, name: 'lead', status: ctx.agents.get(id)?.status || 'inactive', binding: bindings.get(id)},
        ...(await ctx.subagents.listDescendants(id, signal())).map(row => ({...row, sessionId: row.id, name: row.path || row.name || row.id,
          status: ctx.agents.get(row.id)?.status || 'inactive', binding: bindings.get(row.id)}))]};
    }
    if (method === 'GET' && url.pathname === '/commands') return {commands: ctx.commands.list(await agent(text(url.searchParams.get('sessionId'), 'sessionId')))};
    if (method === 'POST' && url.pathname === '/commands') {
      const current = await agent(text(body.sessionId, 'sessionId'));
      const result = await ctx.commands.execute(current, text(body.line, 'line'), [], signal());
      if (!result) throw new HttpError(400, 'Unknown slash command');
      return result;
    }
    if (method === 'POST' && url.pathname === '/roles') {
      if (!ROLES.includes(body.role)) throw new HttpError(400, `role must be ${ROLES.join(', ')}`);
      const parent = await agent(text(body.parentSessionId, 'parentSessionId'));
      const id = `tsad-${body.role}-${randomUUID()}`;
      const binding = {role: body.role, jobId: body.jobId, parentSessionId: parent.id, maxSteps: Number(body.maxSteps || maxSteps)};
      bindings.set(id, binding); await save();
      try {
        const activation = await ctx.subagents.startActivation({provider: 'spawn', childId: id,
          label: body.name || `${body.role}${body.jobId ? `: ${body.jobId}` : ''}`,
          request: {parent, ...(body.cwd ? {cwd: body.cwd} : {}), agentPreset: `tsad-${body.role}`,
            prompt: [{type: 'text', text: text(body.prompt, 'prompt')}], maxDepth: 2},
          signal: signal(), delivery: 'parent'});
        activation.result.catch(error => { errors.set(id, {message: error.message}); journal.append(id, 'session.error', {message: error.message}); });
        return {sessionId: activation.childId, name: body.name || body.role, role: body.role, jobId: body.jobId};
      } catch (error) { bindings.delete(id); await save(); throw error; }
    }
    throw new HttpError(404, 'Unknown bridge endpoint');
  }
  const server = createServer(async (req, res) => {
    res.setHeader('content-type', 'application/json; charset=utf-8');
    res.setHeader('cache-control', 'no-store');
    if (!authorized(req.headers.authorization, token)) { res.writeHead(401); res.end(JSON.stringify({error: {message: 'Unauthorized'}})); return; }
    try {
      const result = await route(req.method, new URL(req.url, 'http://127.0.0.1'), req.method === 'POST' ? await readBody(req) : {});
      res.writeHead(req.method === 'POST' && req.url.endsWith('/messages') ? 202 : 200); res.end(JSON.stringify(result ?? null));
    } catch (error) {
      res.writeHead(error.status || (error.code === 'SESSION_NOT_FOUND' ? 404 : 500));
      res.end(JSON.stringify({error: {message: error.message, code: error.code || 'BRIDGE_ERROR'}}));
    }
  });
  await new Promise((resolveReady, reject) => { server.once('error', reject); server.listen(Number(env.TSAD_BRIDGE_PORT || 8768), '127.0.0.1', resolveReady); });
  ready = true;
  ctx.effect(() => async () => { ready = false; stop.abort(); for (const timer of timers.values()) clearTimeout(timer); server.closeIdleConnections(); await new Promise(done => server.close(done)); await saveChain; });
  ctx.logger.info(`TSAD native bridge listening on 127.0.0.1:${server.address().port}`);
}
