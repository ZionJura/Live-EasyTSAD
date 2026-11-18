/** Exercise the installed published runtime, including preset composition. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createServer as httpServer} from 'node:http';
import {createServer as tcpServer} from 'node:net';
import {existsSync} from 'node:fs';
import {mkdtemp, readFile, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {randomUUID} from 'node:crypto';
import {setTimeout as delay} from 'node:timers/promises';

const hostDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repo = resolve(hostDir, '../../..');
const pinnedNode = join(hostDir, 'node_modules/node/bin/node');
const require = createRequire(import.meta.url);

async function port() {
  const server = tcpServer();
  await new Promise((done, fail) => { server.once('error', fail); server.listen(0, '127.0.0.1', done); });
  const result = server.address().port;
  await new Promise((done, fail) => server.close(error => error ? fail(error) : done()));
  return result;
}

function signalGroup(child, signal) {
  try { process.kill(-child.pid, signal); } catch (error) { if (error.code !== 'ESRCH') throw error; }
}

async function stopHost(child) {
  if (!child?.pid) return;
  // The launcher is a wrapper; its descendants must be stopped as a group.
  signalGroup(child, 'SIGTERM');
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    try { process.kill(-child.pid, 0); } catch (error) { if (error.code === 'ESRCH') return; throw error; }
    await delay(100);
  }
  signalGroup(child, 'SIGKILL');
  if (child.exitCode === null && child.signalCode === null) {
    await Promise.race([new Promise(done => child.once('exit', done)), delay(5000)]);
  }
}

test('published Host persists sessions/Team, mounts commands and creates a real native child with a fixture provider', {timeout: 150000}, async t => {
  let packagePath;
  try { packagePath = require.resolve('@deepseek-ai/dsh/package.json'); }
  catch { t.skip('Published DSH dependencies are absent; run npm ci in dsh-host.'); return; }
  if (!existsSync(pinnedNode)) { t.skip('Pinned Node 24 runtime is absent; run npm ci in dsh-host.'); return; }
  assert.equal(JSON.parse(await readFile(packagePath, 'utf8')).version, '0.2.2-rc.1');
  const home = await mkdtemp(join(tmpdir(), 'tsad-native-runtime-'));
  let modelRequests = 0, fixtureEnabled = false, spawnNext = false, fixtureReply = 'fixture child complete';
  const provider = httpServer((req, res) => {
    modelRequests++;
    if (!fixtureEnabled) { res.writeHead(503); res.end('No model calls are allowed during session and Team controls.'); return; }
    req.resume();
    req.on('end', () => {
      res.writeHead(200, {'content-type': 'text/event-stream'});
      const chunk = (delta, finish_reason = null) => ({id: 'fixture', object: 'chat.completion.chunk', created: 0,
        model: 'deepseek-v4.1-flash', choices: [{index: 0, delta, finish_reason}]});
      if (spawnNext) {
        spawnNext = false;
        res.write(`data: ${JSON.stringify(chunk({role: 'assistant', tool_calls: [{index: 0, id: 'fixture-spawn', type: 'function', function: {
          name: 'subagent', arguments: JSON.stringify({description: 'Native fixture child', prompt: 'Return the fixture answer.'}),
        }}]}))}\n\n`);
        res.write(`data: ${JSON.stringify(chunk({}, 'tool_calls'))}\n\n`);
      } else {
        res.write(`data: ${JSON.stringify(chunk({role: 'assistant', content: fixtureReply}))}\n\n`);
        res.write(`data: ${JSON.stringify(chunk({}, 'stop'))}\n\n`);
      }
      res.end('data: [DONE]\n\n');
    });
  });
  await new Promise(done => provider.listen(0, '127.0.0.1', done));
  const bridgePort = await port();
  const token = randomUUID(), sessionId = `tsad-integration-${randomUUID()}`;
  const marker = `Persistent history ${randomUUID()}`;
  const env = {...process.env, DSH_HOME: home, TSAD_ROOT: repo,
    TSAD_BRIDGE_PORT: String(bridgePort),
    TSAD_BRIDGE_TOKEN: token, TSAD_CONTROL_TOKEN: token,
    TSAD_CONTROL_URL: `http://127.0.0.1:${provider.address().port}`,
    TSAD_MODEL_PROVIDER: 'tsad-integration', TSAD_MODEL_API: 'openai-completions',
    TSAD_MODEL: 'deepseek-v4.1-flash',
    TSAD_MODEL_BASE_URL: `http://127.0.0.1:${provider.address().port}/v1`,
    TSAD_MODEL_API_KEY_ENV: 'TSAD_INTEGRATION_DUMMY_KEY', TSAD_INTEGRATION_DUMMY_KEY: 'dummy-no-requests',
    TSAD_MODEL_PERMITS_DIR: join(home, 'model-permits'), DSH_TELEMETRY: '0'};
  let current, output = '';
  const request = async (path, body) => {
    const response = await fetch(`http://127.0.0.1:${bridgePort}${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: {authorization: `Bearer ${token}`, 'content-type': 'application/json'},
      ...(body === undefined ? {} : {body: JSON.stringify(body)}), signal: AbortSignal.timeout(10000),
    });
    const data = await response.json();
    assert.equal(response.ok, true, `${path}: HTTP ${response.status} ${JSON.stringify(data)}`);
    return data;
  };
  const boot = async () => {
    output = '';
    current = spawn(pinnedNode, [join(hostDir, 'launch.mjs')], {cwd: repo, env, detached: true, stdio: ['ignore', 'pipe', 'pipe']});
    let spawnError;
    current.on('error', error => { spawnError = error; });
    const append = data => { output = (output + data.toString()).slice(-24000); };
    current.stdout.on('data', append); current.stderr.on('data', append);
    const deadline = Date.now() + 60000;
    while (Date.now() < deadline) {
      if (spawnError) throw spawnError;
      if (current.exitCode !== null || current.signalCode !== null) throw new Error(`Published Host exited before readiness:\n${output}`);
      try {
        const health = await request('/health');
        if (health.ready) {
          assert.equal(health.version, '0.2.2-rc.1');
          assert.equal(health.teamImplementation, 'tsad-adapter');
          return;
        }
      } catch (error) { if (error.name === 'AssertionError') throw error; }
      await delay(150);
    }
    throw new Error(`Published Host did not become ready within 60 seconds:\n${output}`);
  };
  try {
    await boot();
    const manifest = JSON.parse(await readFile(join(home, 'profiles/tsad-interactive/package.json'), 'utf8'));
    assert.deepEqual(manifest.dsh.profile.bundles, ['@deepseek-ai/dsh-base']);
    assert(!output.includes('webserver listening'), output);
    const created = await request('/sessions', {sessionId, cwd: repo, systemPrompt: marker});
    assert.equal(created.sessionId, sessionId);
    const first = await request(`/sessions/${sessionId}`);
    assert.equal(first.sessionId, sessionId);
    assert.equal(first.running, false);
    const commands = await request(`/commands?sessionId=${encodeURIComponent(sessionId)}`);
    for (const name of ['goal', 'compact', 'plan']) assert(commands.commands.some(command => command.name === name), `Missing native /${name} command`);
    const task = await request(`/sessions/${sessionId}/team`, {action: 'create', task: {subject: 'Inspect official source', description: marker}});
    const updated = await request(`/sessions/${sessionId}/team`, {action: 'update', task: {taskId: task.id, expectedRevision: task.revision, action: 'complete'}});
    assert.equal(updated.status, 'completed');
    const before = await request(`/sessions/${sessionId}`);
    assert.equal(before.team.tasks[0].status, 'completed');
    // inject() queues setup text without waking a turn; no chat is fabricated.
    assert.deepEqual(before.history, []);
    assert.equal(before.binding.systemPrompt, marker);
    await stopHost(current); current = null;
    await boot();
    const after = await request(`/sessions/${sessionId}`);
    assert.equal(after.sessionId, sessionId);
    assert.deepEqual(after.team, before.team);
    assert.deepEqual(after.history, before.history);
    assert.equal(after.binding.systemPrompt, marker);
    assert.equal(modelRequests, 0, 'Session, command and Team controls must not request a model');
    fixtureEnabled = true;
    const child = await request('/roles', {parentSessionId: sessionId, role: 'researcher', name: 'fixture-researcher', prompt: 'Return the fixture answer.'});
    assert.equal(child.role, 'researcher');
    assert.equal(child.name, 'fixture-researcher');
    const childDeadline = Date.now() + 20000;
    let childSnapshot;
    while (Date.now() < childDeadline) {
      childSnapshot = await request(`/sessions/${child.sessionId}`);
      if (!childSnapshot.running && JSON.stringify(childSnapshot.history).includes('fixture child complete')) break;
      await delay(100);
    }
    assert.equal(childSnapshot.binding.parentSessionId, sessionId);
    assert.equal(childSnapshot.binding.role, 'researcher');
    assert.equal(childSnapshot.lastError, null);
    assert(JSON.stringify(childSnapshot.history).includes('fixture child complete'), 'Native child did not finish through the local fixture provider');
    assert(modelRequests > 0, 'Native role creation did not execute its real child');

    // Unlike /roles, a model-created child only receives the agent/created
    // inherited binding. Its quiescent Session must remain addressable.
    spawnNext = true;
    await request(`/sessions/${sessionId}/messages`, {text: 'Delegate the fixture task.', requestId: randomUUID()});
    let nativeChild, nativeSnapshot;
    const nativeDeadline = Date.now() + 20000;
    while (Date.now() < nativeDeadline) {
      const rows = (await request(`/agents?sessionId=${encodeURIComponent(sessionId)}`)).agents;
      nativeChild = rows.find(row => row.id !== sessionId && row.id !== child.sessionId);
      if (nativeChild) {
        nativeSnapshot = await request(`/sessions/${nativeChild.id}`);
        if (!nativeSnapshot.running && JSON.stringify(nativeSnapshot.history).includes('fixture child complete')) break;
      }
      await delay(100);
    }
    assert(nativeChild, 'The native model-facing subagent tool did not create a child');
    assert(JSON.stringify(nativeSnapshot?.history).includes('fixture child complete'));
    assert.equal(nativeSnapshot.binding.inherited, true);
    assert.equal(nativeSnapshot.binding.parentSessionId, sessionId, 'Inherited binding must record the direct parent');
    assert.equal(nativeSnapshot.binding.role, 'lead');

    fixtureReply = 'fixture child resumed live';
    const liveResume = await request(`/sessions/${nativeChild.id}/messages`, {text: 'Resume the fixture child.', requestId: randomUUID()});
    assert(liveResume.messageId, 'Native child follow-up must return an accepted inbox identity');
    let resumed;
    const liveResumeDeadline = Date.now() + 20000;
    while (Date.now() < liveResumeDeadline) {
      resumed = await request(`/sessions/${nativeChild.id}`);
      if (!resumed.running && JSON.stringify(resumed.history).includes(fixtureReply)) break;
      await delay(100);
    }
    assert(JSON.stringify(resumed.history).includes(fixtureReply), 'Completed native child did not accept a user follow-up');
    assert.deepEqual(await request(`/sessions/${nativeChild.id}/cancel`, {}), {accepted: true});

    // Reproduce persisted pre-fix bindings and an empty live registry. The
    // runtime validates direct ownership against its own durable header and
    // subagent projection; no synthetic role binding may be required to read.
    await stopHost(current); current = null;
    const bindingsPath = join(home, 'tsad-role-bindings.json');
    const legacy = JSON.parse(await readFile(bindingsPath, 'utf8'));
    delete legacy[nativeChild.id].parentSessionId;
    await writeFile(bindingsPath, JSON.stringify(legacy));
    const requestsBeforeRead = modelRequests;
    await boot();
    const restoredNative = await request(`/sessions/${nativeChild.id}`);
    assert.equal(restoredNative.binding.parentSessionId, undefined);
    assert.deepEqual(restoredNative.history, resumed.history);
    assert.equal(modelRequests, requestsBeforeRead, 'Reading a durable child must not activate a model');
    const inactiveControls = await fetch(`http://127.0.0.1:${bridgePort}/commands?sessionId=${nativeChild.id}`, {
      headers: {authorization: `Bearer ${token}`}, signal: AbortSignal.timeout(10000),
    });
    assert.equal(inactiveControls.status, 409);
    const inactiveError = await inactiveControls.json();
    assert(inactiveError.error.message.includes(`direct parent ${sessionId}`));
    assert(inactiveError.error.message.includes('no restore-only subagent API'));
    assert.equal(modelRequests, requestsBeforeRead, 'Inactive controls must not synthesize a prompt to restore ancestry');
    fixtureReply = 'fixture child resumed after restart';
    const coldResume = await request(`/sessions/${nativeChild.id}/messages`, {text: 'Resume after restarting the Host.', mode: 'steer', requestId: randomUUID()});
    assert(coldResume.messageId);
    const coldResumeDeadline = Date.now() + 20000;
    let coldSnapshot;
    while (Date.now() < coldResumeDeadline) {
      coldSnapshot = await request(`/sessions/${nativeChild.id}`);
      if (!coldSnapshot.running && JSON.stringify(coldSnapshot.history).includes(fixtureReply)) break;
      await delay(100);
    }
    assert(JSON.stringify(coldSnapshot.history).includes(fixtureReply), 'Durable child must cold-resume through its restored direct parent');
  } catch (error) {
    t.diagnostic(output);
    throw error;
  } finally {
    await stopHost(current);
    provider.closeAllConnections();
    await new Promise(done => provider.close(done));
    await rm(home, {recursive: true, force: true});
  }
});
