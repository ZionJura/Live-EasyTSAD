import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, readFile, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {TeamAdapter} from '../src/team-adapter.mjs';

const lead = {id: 'lead'}, child = {id: 'child'}, other = {id: 'other'};
async function board(t) {
  const home = await mkdtemp(join(tmpdir(), 'tsad-team-'));
  t.after(() => rm(home, {recursive: true, force: true}));
  return {home, teams: await TeamAdapter.open(home, () => 'lead')};
}

test('task board persists across restart and shares its root with descendants', async t => {
  const {home, teams} = await board(t);
  const task = await teams.createTask(lead, {subject: 'Read official source', description: 'Verify the interface'});
  assert.equal(task.status, 'pending');
  assert.deepEqual(teams.listTasks(child), teams.listTasks(lead));
  await teams.setEnabled(lead, false);
  const restored = await TeamAdapter.open(home, () => 'lead');
  assert.equal(restored.snapshot(lead).enabled, false);
  assert.equal(restored.snapshot(lead).implementation, 'tsad-adapter');
  assert.equal(restored.listTasks(lead)[0].subject, task.subject);
  await assert.rejects(restored.createTask(lead, {subject: 'blocked'}), /disabled/);
  await assert.rejects(restored.setEnabled(child, true), /root session/);
  assert.equal(JSON.parse(await readFile(restored.path, 'utf8')).version, 1);
});

test('serialized CAS admits one concurrent update and preserves the committed revision', async t => {
  const {teams} = await board(t);
  const task = await teams.createTask(lead, {subject: 'Inspect'});
  const updates = await Promise.allSettled([
    teams.updateTask(lead, {taskId: task.id, expectedRevision: 1, action: 'claim'}),
    teams.updateTask(lead, {taskId: task.id, expectedRevision: 1, action: 'edit', subject: 'stale title'}),
  ]);
  assert.equal(updates.filter(result => result.status === 'fulfilled').length, 1);
  assert.match(updates.find(result => result.status === 'rejected').reason.message, /revision changed/);
  assert.equal(teams.listTasks(lead)[0].revision, 2);
  assert.equal(teams.listTasks(lead)[0].subject, 'Inspect');
});

test('dependencies reject missing targets, cycles, premature claims and referenced deletion', async t => {
  const {teams} = await board(t);
  await assert.rejects(teams.createTask(lead, {subject: 'Invalid', blockedBy: ['missing']}), /does not exist/);
  const first = await teams.createTask(lead, {subject: 'Source'});
  const second = await teams.createTask(lead, {subject: 'Adapter', blockedBy: [first.id]});
  assert.equal(first.id, '1');
  assert.equal(second.ready, false);
  await assert.rejects(teams.updateTask(lead, {taskId: second.id, expectedRevision: 1, action: 'claim'}), /not ready/);
  await assert.rejects(teams.updateTask(lead, {taskId: first.id, expectedRevision: 1, action: 'set_dependencies', blockedBy: [second.id]}), /cycle/);
  assert.equal(teams.listTasks(lead)[0].revision, 1);
  await assert.rejects(teams.updateTask(lead, {taskId: first.id, expectedRevision: 1, action: 'delete'}), /does not exist/);
  await teams.updateTask(lead, {taskId: first.id, expectedRevision: 1, action: 'complete'});
  assert.equal(teams.listTasks(lead)[1].ready, true);
  await teams.updateTask(child, {taskId: second.id, expectedRevision: 1, action: 'claim'});
  await assert.rejects(teams.updateTask(other, {taskId: second.id, expectedRevision: 2, action: 'complete'}), /another agent/);
  await teams.updateTask(child, {taskId: second.id, expectedRevision: 2, action: 'complete'});
});

test('invalid ids, arbitrary status writes, corrupt state and the 256-task bound are rejected', async t => {
  const {home, teams} = await board(t);
  await assert.rejects(teams.createTask(lead, {subject: 'Invalid', status: 'completed'}), /Unsupported Team field/);
  await assert.rejects(teams.updateTask(lead, {taskId: '../x', expectedRevision: 1, action: 'complete'}), /task id/);
  const task = {revision: 1, subject: 'Task', description: '', status: 'pending', blockedBy: [], writeScopes: []};
  const tasks = Array.from({length: 256}, (_, index) => ({...task, id: String(index + 1)}));
  await writeFile(teams.path, JSON.stringify({version: 1, boards: {lead: {enabled: true, nextId: 257, tasks}}}));
  const full = await TeamAdapter.open(home);
  await assert.rejects(full.createTask(lead, {subject: 'Overflow'}), /256/);
  tasks[0].status = 'invented';
  await writeFile(teams.path, JSON.stringify({version: 1, boards: {lead: {enabled: true, nextId: 257, tasks}}}));
  await assert.rejects(TeamAdapter.open(home), /Invalid Team task state/);
});
