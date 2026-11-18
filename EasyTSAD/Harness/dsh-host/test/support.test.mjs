import test from 'node:test';
import assert from 'node:assert/strict';
import {EventJournal, Semaphore, authorized, roleGuard} from '../src/support.mjs';
import {buildPatch} from '../src/profile.mjs';

test('bridge bearer credential is required and compared exactly', () => {
  assert.equal(authorized(undefined, 'a'.repeat(20)), false);
  assert.equal(authorized('Bearer ' + 'a'.repeat(20), 'a'.repeat(20)), true);
  assert.equal(authorized('Bearer ' + 'b'.repeat(20), 'a'.repeat(20)), false);
  assert.equal(authorized('Bearer ', ''), false);
});

test('read-only supervisor cannot mutate files, spawn or control training', () => {
  const binding = {role: 'supervisor', jobId: 'job'};
  for (const name of ['bash', 'pwsh', 'write', 'edit', 'subagent', 'run_code', 'tsad_run', 'tsad_stop', 'tsad_correct', 'tsad_environment']) assert.ok(roleGuard(binding, name), name);
  for (const name of ['read', 'grep', 'web_fetch', 'tsad_scores', 'tsad_diagnose', 'tsad_status']) assert.equal(roleGuard(binding, name), undefined, name);
});

test('coordinator and implementer cannot acquire unrestricted shell through another preset', () => {
  assert.ok(roleGuard({role: 'lead'}, 'bash'));
  assert.ok(roleGuard({role: 'lead'}, 'write'));
  assert.ok(roleGuard({role: 'implementer', jobId: 'job'}, 'bash'));
  assert.ok(roleGuard({role: 'implementer'}, 'write'));
  assert.equal(roleGuard({role: 'implementer', jobId: 'job'}, 'write'), undefined);
  assert.ok(roleGuard({role: 'lead', jobId: 'job'}, 'update_goal', {status: 'complete'}));
});

test('bounded journal separates sessions and detects dropped events or restart cursor', () => {
  const journal = new EventJournal(2);
  journal.append('a', 'one', {}); journal.append('b', 'two', {}); journal.append('a', 'three', {});
  assert.deepEqual(journal.read('a', 1).events.map(e => e.type), ['three']);
  assert.equal(journal.read('a', 20).reset, true);
  journal.append('a', 'four', {});
  assert.equal(journal.read('a', 1).reset, true);
});

test('model stream semaphore enforces concurrency and aborts queued waiters', async () => {
  const semaphore = new Semaphore(1), abort = new AbortController();
  const release = await semaphore.acquire();
  const waiting = semaphore.acquire(abort.signal);
  abort.abort(new Error('cancelled'));
  await assert.rejects(waiting, /cancelled/);
  release(); release();
  const next = await semaphore.acquire();
  assert.equal(semaphore.active, 1); next(); assert.equal(semaphore.active, 0);
});

test('profile pins roles to native capabilities and does not enable workflow or telemetry', () => {
  const patch = buildPatch({}, '/tmp/host');
  assert.deepEqual(patch.find(row => row.id === 'subagent').config, {maxDepth: 2, maxActiveSubagents: 8});
  const presets = patch.at(-1).insert.filter(row => row.id.startsWith('tsad-preset'));
  const children = preset => preset.config.plugins[0].config;
  for (const preset of presets) {
    assert.equal(preset.config.plugins[0].name, 'cordis:group');
    assert.deepEqual(preset.config.plugins[0].isolate, {compaction: true, planMode: true});
    assert.equal(children(preset).some(row => row.name.endsWith('tool-bash')), false);
  }
  const fs = children(presets.find(row => row.id === 'tsad-preset-supervisor')).find(row => row.id === 'tool-fs').config;
  assert.equal(fs.write, false); assert.equal(fs.edit, false);
  assert.equal(patch.find(row => row.id === 'ptc-runtime').disabled, true);
  assert.equal(patch.some(row => row.id === 'agent-team'), false);
  assert.ok(children(presets.find(row => row.id === 'tsad-preset-lead')).find(row => row.id === 'plan-mode').config.section.trim());
});

test('local Team tool is restricted to lead sessions', () => {
  assert.equal(roleGuard({role: 'lead'}, 'tsad_team'), undefined);
  for (const role of ['researcher', 'implementer', 'supervisor', 'reporter']) assert.ok(roleGuard({role}, 'tsad_team'));
  assert.ok(roleGuard(undefined, 'tsad_team'));
});


test('terminal profile mounts only native backend services without browser plugins', () => {
  const patch = buildPatch({}, '/tmp/host');
  const inserted = patch.flatMap(row => row.insert ?? []);
  for (const name of ['session-controller', 'agent-preset-registry', 'workspace', 'connection', 'file-upload']) {
    assert(inserted.some(row => row.id === name), `Missing required native backend ${name}`);
  }
  for (const row of inserted) {
    assert(!/dsh-(?:web-app|host-webserver|client-ui-|client-modules|client-hmr)/.test(row.name), row.name);
  }
});
