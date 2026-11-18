import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer, type ServerResponse} from 'node:http';
import {mkdtemp, mkdir, writeFile, rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {RemoteHarnessAdapter, normalizeHistory, streamDelta, type RemoteMessage} from '../src/adapter.ts';

type Row = Record<string, unknown>;
async function fixture(run: (server: {url: string; calls: {method: string; path: string; body: Row}[]; reply: (handler: (path: string, response: ServerResponse) => boolean) => void}) => Promise<void>) {
  const calls: {method: string; path: string; body: Row}[] = [];
  const streams = new Set<ServerResponse>();
  let handler: (path: string, response: ServerResponse) => boolean = () => false;
  const server = createServer(async (request, response) => {
    let raw = ''; for await (const chunk of request) raw += String(chunk);
    const body = raw ? JSON.parse(raw) as Row : {};
    const path = (request.url || '').replace('/api/harness/v1', '');
    calls.push({method: request.method || 'GET', path, body});
    if (request.headers.authorization !== 'Bearer fixture-token') { response.writeHead(401); response.end('{}'); return; }
    if (path === '/events') { response.writeHead(200, {'Content-Type': 'text/event-stream'}); response.write(': alive\n\n'); streams.add(response); response.on('close', () => streams.delete(response)); return; }
    response.setHeader('Content-Type', 'application/json');
    if (handler(path, response)) return;
    let result: unknown = {};
    if (path === '/health') result = {ok: true, connected: true};
    else if (path === '/sessions') result = request.method === 'POST' ? {sessionId: 'new-root'} : {sessions: [{sessionId: 'root'}]};
    else if (path.includes('/events?')) result = {events: [], cursor: 0, epoch: 'host-1'};
    else if (path.startsWith('/sessions/') && request.method === 'GET') result = {sessionId: path.split('/')[2], status: 'idle', history: []};
    else if (path.startsWith('/commands?')) result = {commands: [{name: 'ps', description: 'processes'}]};
    else if (path === '/commands') result = {processes: [{pid: 123, status: 'running'}]};
    else if (path === '/batches') result = {batches: []};
    response.end(JSON.stringify(result));
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address(); assert.ok(address && typeof address === 'object');
  try { await run({url: `http://127.0.0.1:${address.port}`, calls, reply: next => { handler = next; }}); }
  finally { for (const response of streams) response.end(); server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())); }
}
function output() { const messages: RemoteMessage[][] = [], status: string[] = [], notices: string[] = []; return {messages, status, notices,
  onMessages: (value: RemoteMessage[]) => { messages.push(value); }, onStatus: (value: string) => { status.push(value); }, onNotice: (value: string) => { notices.push(value); }}; }
const json = (response: ServerResponse, value: unknown) => { response.end(JSON.stringify(value)); return true; };

test('native history retains tool blocks and call ids and applies replacement surfaces', () => {
  const call = {type: 'tool-call', id: 'call-1', name: 'read', arguments: {path: 'config.toml'}};
  const messages = normalizeHistory({historyPage: {records: [
    {type: 'event', event: {seq: 1, type: 'user/message', data: {role: 'user', content: 'hello'}}},
    {type: 'event', event: {seq: 2, type: 'assistant/message', data: {message: {role: 'assistant', content: 'superseded'}}}},
    {type: 'event', event: {seq: 3, type: 'assistant/message', data: {message: {role: 'assistant', content: [call]}}, surfaceOp: {op: 'replace', startSeq: 2, endSeq: 2}}},
    {type: 'event', event: {seq: 4, type: 'tool/result', data: {message: {role: 'tool', callId: 'call-1', content: [{type: 'text', text: 'file contents'}]}}}},
  ]}});
  assert.equal(messages.length, 3); assert.deepEqual(messages[1].content, [call]); assert.equal(messages[2].toolCallId, 'call-1');
  assert.equal(streamDelta({frame: {chunk: {type: 'text-delta', text: '实时'}}}), '实时');
  assert.equal(streamDelta({chunk: {type: 'thinking-delta', text: 'hidden'}}), '');
  const cli = normalizeHistory({transport: 'cli', events: [
    {type: 'text', text: 'checking'}, {type: 'tool_call', callId: 'c2', tool: 'read', input: {path: 'run.py'}},
    {type: 'tool_result', callId: 'c2', result: 'source'}, {type: 'text', text: 'done'}, {type: 'final', text: 'done'},
  ], history: [{role: 'assistant', content: 'done'}]});
  assert.equal(cli.filter(message => message.content === 'done').length, 1);
  assert.equal(cli[2].toolCallId, 'c2');
  assert.deepEqual(cli[1].content, [{type: 'tool-call', id: 'c2', name: 'read', arguments: {path: 'run.py'}}]);
});

test('native stream final commit replaces live output without duplicates; commands and gpu use domain API', async () => fixture(async ({url, reply, calls}) => {
  let step = 0;
  reply((path, response) => {
    if (path.startsWith('/sessions/root/events?')) {
      step++;
      if (step === 1) return json(response, {events: [], cursor: 0, epoch: 'epoch-1'});
      if (step === 2) return json(response, {events: [{seq: 1, sessionId: 'root', type: 'assistant.stream', data: {chunk: {type: 'text-delta', text: 'answer'}}}], cursor: 1, epoch: 'epoch-1'});
      return json(response, {events: [{seq: 2, sessionId: 'root', type: 'session.event', data: {seq: 5, type: 'assistant/message', data: {message: {role: 'assistant', content: 'answer'}}}}], cursor: 2, epoch: 'epoch-1'});
    }
    if (path === '/sessions/root') return json(response, {running: true, status: 'running', history: step >= 3 ? [{id: 'saved', role: 'assistant', content: 'answer'}] : []});
    return false;
  });
  const sink = output(), adapter = new RemoteHarnessAdapter({resume: 'latest', url, token: 'fixture-token', refresh: 600, gpus: '0,1', task: 'job-1', ...sink});
  try {
    await adapter.connect(); await adapter.poll();
    assert.deepEqual(sink.messages.at(-1)?.map(message => [message.content, message.streaming]), [['answer', true]]);
    await adapter.poll(); assert.deepEqual(sink.messages.at(-1)?.map(message => message.content), ['answer']);
    await adapter.command('/ps'); assert.match(String(sink.messages.at(-1)?.at(-1)?.content), /123/);
    await adapter.command('/run'); assert.ok(calls.some(call => call.path === '/jobs/job-1/run' && call.body.gpus === '0,1'));
    await adapter.send('补充说明', {mode: 'steer'});
    const message = calls.find(call => call.path === '/sessions/root/messages'); assert.equal(message?.body.mode, 'steer'); assert.ok(message?.body.requestId);
    await adapter.cancel(); assert.ok(calls.some(call => call.path === '/sessions/root/cancel'));
  } finally { adapter.dispose(); }
}));

test('CLI final text arriving after terminal event is recovered by periodic snapshots', async () => fixture(async ({url, reply}) => {
  let snapshot = 0, poll = 0;
  reply((path, response) => {
    if (path.startsWith('/sessions/root/events?')) {
      poll++; return json(response, {events: poll === 1 ? [{type: 'text', text: 'partial'}, {type: 'final', text: 'full answer'}] : [], cursor: 2, epoch: 'cli-log'});
    }
    if (path === '/sessions/root') {
      snapshot++; return json(response, {transport: 'cli', eventEpoch: 'cli-log', running: false, status: 'done', history: snapshot < 2 ? [] : [{role: 'assistant', content: 'full answer'}]});
    }
    return false;
  });
  const sink = output(), adapter = new RemoteHarnessAdapter({resume: 'latest', url, token: 'fixture-token', refresh: .1, ...sink});
  try {
    await adapter.connect();
    assert.ok(sink.messages.at(-1)?.some(message => message.content === 'full answer'));
    await new Promise(resolve => setTimeout(resolve, 650)); await adapter.poll();
    assert.equal(sink.messages.at(-1)?.filter(message => message.content === 'full answer').length, 1);
    assert.equal(sink.messages.at(-1)?.some(message => message.streaming), false);
  } finally { adapter.dispose(); }
}));

test('switching sessions fences slow in-flight responses and disposal only closes terminal I/O', async () => fixture(async ({url, reply, calls}) => {
  let delayed = false, release: (() => void) | undefined;
  reply((path, response) => {
    if (path === '/sessions/root') {
      if (delayed) { release = () => json(response, {history: [{role: 'assistant', content: 'stale root'}]}); return true; }
      return json(response, {history: [{role: 'assistant', content: 'root'}]});
    }
    if (path === '/sessions/child') return json(response, {history: [{role: 'assistant', content: 'child answer'}]});
    return false;
  });
  const sink = output(), adapter = new RemoteHarnessAdapter({resume: 'latest', url, token: 'fixture-token', refresh: 600, ...sink});
  try {
    await adapter.connect(); delayed = true;
    const reconnect = adapter.command('/ps');
    await new Promise(resolve => setTimeout(resolve, 25));
    await adapter.command('/agent child'); release?.(); await reconnect;
    await new Promise(resolve => setTimeout(resolve, 30)); await adapter.poll();
    assert.equal(adapter.selectedSessionId, 'child'); assert.equal(sink.messages.at(-1)?.some(message => message.content === 'stale root'), false);
    assert.ok(sink.messages.at(-1)?.some(message => message.content === 'child answer'));
    adapter.dispose(); assert.equal(calls.some(call => /shutdown|\/stop/.test(call.path)), false);
  } finally { adapter.dispose(); }
}));

test('changed runtime epoch refreshes canonical history instead of appending old streamed text', async () => fixture(async ({url, reply}) => {
  let epoch = 'first';
  reply((path, response) => {
    if (path.includes('/events?')) return json(response, {events: [], cursor: 0, epoch, reset: epoch === 'second'});
    if (path === '/sessions/root') return json(response, {history: [{role: 'assistant', content: epoch}]});
    return false;
  });
  const sink = output(), adapter = new RemoteHarnessAdapter({resume: 'latest', url, token: 'fixture-token', refresh: 600, ...sink});
  try { await adapter.connect(); epoch = 'second'; await adapter.poll(); assert.deepEqual(sink.messages.at(-1)?.map(message => message.content), ['second']); }
  finally { adapter.dispose(); }
}));

test('history paging merges by durable seq and newer compaction replacements remove older messages', async () => fixture(async ({url, reply}) => {
  let compacted = false;
  const raw = (seq: number, type: string, data: unknown, surfaceOp?: unknown) => ({type: 'event', event: {seq, type, data, ...(surfaceOp ? {surfaceOp} : {})}});
  reply((path, response) => {
    if (path.includes('/events?')) return json(response, {events: compacted ? [{type: 'session.event', data: {type: 'assistant/message'}}] : [], cursor: compacted ? 1 : 0, epoch: 'same'});
    if (path.startsWith('/sessions/root?beforeSeq=3')) return json(response, {historyPage: {hasMore: false, records: [
      raw(1, 'user/message', {role: 'user', content: 'old question'}), raw(2, 'assistant/message', {message: {role: 'assistant', content: 'old response'}}),
    ]}});
    if (path === '/sessions/root') return json(response, {running: false, historyPage: {hasMore: true, records: compacted ? [
      raw(3, 'user/message', {role: 'user', content: 'new question'}),
      raw(4, 'assistant/message', {message: {role: 'assistant', content: 'corrected old response'}}, {op: 'replace', startSeq: 2, endSeq: 2}),
    ] : [raw(3, 'user/message', {role: 'user', content: 'new question'})]}});
    return false;
  });
  const sink = output(), adapter = new RemoteHarnessAdapter({resume: 'latest', url, token: 'fixture-token', refresh: 600, ...sink});
  try {
    await adapter.connect(); await adapter.command('/history older');
    assert.deepEqual(sink.messages.at(-1)?.map(message => message.content), ['old question', 'old response', 'new question']);
    compacted = true; await adapter.poll();
    assert.deepEqual(sink.messages.at(-1)?.map(message => message.content), ['old question', 'corrected old response', 'new question']);
    await adapter.command('/history'); assert.match(sink.notices.at(-1) || '', /全部/);
  } finally { adapter.dispose(); }
}));

test('independent clients attach to the same service without starting task runs or sharing streams', async () => fixture(async ({url, calls}) => {
  const a = output(), b = output();
  const first = new RemoteHarnessAdapter({resume: 'latest', url, token: 'fixture-token', task: 'existing-job', refresh: 600, ...a});
  const second = new RemoteHarnessAdapter({resume: 'latest', url, token: 'fixture-token', task: 'existing-job', refresh: 600, ...b});
  try {
    await Promise.all([first.connect(), second.connect()]);
    assert.equal(calls.some(call => call.method === 'POST'), false);
    await first.command('/agent another');
    assert.equal(first.selectedSessionId, 'another'); assert.equal(second.selectedSessionId, 'root');
    first.dispose(); await second.send('still connected');
    assert.ok(calls.some(call => call.path === '/sessions/root/messages'));
    assert.equal(calls.some(call => /\/run|\/stop|shutdown/.test(call.path)), false);
  } finally { first.dispose(); second.dispose(); }
}));

test('a reconnect recovers after transport failure and refreshes the active session without replaying messages', async () => fixture(async ({url, calls, reply}) => {
  let fail = false;
  reply((path, response) => {
    if (path.includes('/events?') && fail) { response.writeHead(503); response.end('{"error":"runtime restarting"}'); return true; }
    if (path === '/sessions/root') return json(response, {status: 'idle', history: [{role: 'assistant', content: fail ? 'old' : 'recovered'}]});
    return false;
  });
  const sink = output(), adapter = new RemoteHarnessAdapter({resume: 'latest', url, token: 'fixture-token', refresh: 600, ...sink});
  try {
    await adapter.connect(); fail = true; await adapter.poll(); assert.match(sink.status.at(-1) || '', /暂不可用/);
    fail = false; await adapter.command('/connect');
    assert.ok(sink.messages.at(-1)?.some(message => message.content === 'recovered'));
    assert.equal(calls.some(call => call.method === 'POST'), false);
  } finally { adapter.dispose(); }
}));


test('custom TOML port and runtime token file are resolved before probing any default service', async () => fixture(async ({url, calls}) => {
  const root = await mkdtemp(join(tmpdir(), 'tsad-tui-config-'));
  await mkdir(join(root, 'EasyTSAD/Harness'), {recursive: true});
  await mkdir(join(root, 'custom-runtime'), {recursive: true});
  await writeFile(join(root, 'EasyTSAD/Harness/config.toml'), `[interactive]\nport = ${new URL(url).port}\nruntime_dir = "custom-runtime"\n`);
  await writeFile(join(root, 'custom-runtime/access-token'), 'fixture-token\n');
  const sink = output(), adapter = new RemoteHarnessAdapter({resume: 'latest', root, refresh: 600, ...sink});
  try { await adapter.connect(); assert.equal(adapter.selectedSessionId, 'root'); assert.ok(calls.some(call => call.path === '/health')); }
  finally { adapter.dispose(); await rm(root, {recursive: true, force: true}); }
}));

test('invalid connection configuration fails visibly without starting the canonical helper', async () => {
  const root = await mkdtemp(join(tmpdir(), 'tsad-tui-bad-config-'));
  await mkdir(join(root, 'EasyTSAD/Harness'), {recursive: true});
  await writeFile(join(root, 'EasyTSAD/Harness/config.toml'), '[interactive]\nport = "wrong"\n');
  const adapter = new RemoteHarnessAdapter({resume: 'latest', root, python: '/missing/helper-must-not-run', ...output()});
  try { await assert.rejects(adapter.connect(), /interactive.port/); }
  finally { adapter.dispose(); await rm(root, {recursive: true, force: true}); }
});

test('default clients request independent sessions without restoring shared history', async () => fixture(async ({url, calls}) => {
  const a = new RemoteHarnessAdapter({url, token: 'fixture-token', refresh: 600, ...output()});
  const b = new RemoteHarnessAdapter({url, token: 'fixture-token', refresh: 600, ...output()});
  try {
    await Promise.all([a.connect(), b.connect()]);
    const creates = calls.filter(call => call.path === '/sessions' && call.method === 'POST');
    assert.equal(creates.length, 2);
    assert.notEqual(creates[0].body.sessionId, creates[1].body.sessionId);
    assert.equal(calls.some(call => call.path === '/sessions' && call.method === 'GET'), false);
  } finally { a.dispose(); b.dispose(); }
}));

test('concurrent creations never replace a later explicit selection', async () => fixture(async ({url, reply}) => {
  const pending: ServerResponse[] = [];
  reply((path, response) => { if (path === '/sessions') { pending.push(response); return true; } return false; });
  const adapter = new RemoteHarnessAdapter({resume: 'root', url, token: 'fixture-token', refresh: 600, ...output()});
  try {
    await adapter.connect();
    const requests = [adapter.command('/new --background'), adapter.command('/new --background'), adapter.command('/new')];
    for (let i = 0; pending.length < 3 && i < 100; i++) await new Promise(resolve => setTimeout(resolve, 5));
    assert.equal(pending.length, 3);
    await adapter.select('user-selected');
    pending.reverse().forEach((response, i) => json(response, {sessionId: `created-${i}`}));
    await Promise.all(requests);
    assert.equal(adapter.selectedSessionId, 'user-selected');
    assert.equal((adapter.getState().creations as Row[]).filter(row => row.status === 'created').length, 3);
  } finally { adapter.dispose(); }
}));

test('slow commands do not hold connection or history rendering', async () => fixture(async ({url, reply}) => {
  let commands: ServerResponse | undefined;
  reply((path, response) => {
    if (path.startsWith('/commands?')) { commands = response; return true; }
    if (path === '/sessions/root') return json(response, {history: [{role: 'assistant', content: 'history ready'}]});
    return false;
  });
  const sink = output(), adapter = new RemoteHarnessAdapter({resume: 'root', url, token: 'fixture-token', refresh: 600, ...sink});
  try {
    await Promise.race([adapter.connect(), new Promise((_, reject) => setTimeout(() => reject(new Error('commands blocked connection')), 1000).unref())]);
    assert.ok(commands);
    assert.equal(adapter.getState().connected, true);
    assert.ok(sink.status.some(value => value.includes('后台已连接')));
    assert.ok(sink.messages.at(-1)?.some(value => value.content === 'history ready'));
    json(commands, {commands: []});
  } finally { adapter.dispose(); }
}));

test('latest resume chooses newest activity and session errors preserve service connectivity', async () => fixture(async ({url, reply}) => {
  let fail = false;
  reply((path, response) => {
    if (path === '/sessions') return json(response, {sessions: [{sessionId: 'newest', updatedAt: 30}, {sessionId: 'oldest', updatedAt: 1}]});
    if (fail && path.includes('/events?')) { response.writeHead(409); return json(response, {error: 'inactive child'}); }
    return false;
  });
  const adapter = new RemoteHarnessAdapter({resume: 'latest', url, token: 'fixture-token', refresh: 600, ...output()});
  try {
    await adapter.connect(); assert.equal(adapter.selectedSessionId, 'newest');
    fail = true; await adapter.poll(); assert.equal(adapter.getState().connected, true);
  } finally { adapter.dispose(); }
}));
