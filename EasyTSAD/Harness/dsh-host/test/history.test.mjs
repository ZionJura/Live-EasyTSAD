import test from 'node:test';
import assert from 'node:assert/strict';
import {sessionHistory} from '../src/history.mjs';

const record = (seq, type, data, surfaceOp = 'append') => ({type: 'event', event: {seq, type, time: 1, data, surfaceOp}});

test('native SessionPage envelopes expose user and nested assistant messages', () => {
  const user = {role: 'user', content: [{type: 'text', text: 'hello'}]};
  const assistant = {role: 'assistant', content: [{type: 'text', text: 'saved answer'}]};
  const history = sessionHistory({cursor: 3, records: [
    record(1, 'user/message', user), record(2, 'assistant/message', {turn: 1, step: 1, message: assistant, stream: []}),
    record(3, 'turn/end', {reason: 'completed'}),
  ], hasMore: false});
  assert.deepEqual(history.map(({role, content}) => ({role, content})), [user, assistant]);
  assert.deepEqual(history.map(({seq}) => seq), [1, 2]);
});

test('surface replacements remove superseded messages at their original position', () => {
  const history = sessionHistory({records: [
    record(1, 'user/message', {role: 'user', content: 'question'}),
    record(2, 'assistant/message', {message: {role: 'assistant', content: 'old answer'}}),
    record(3, 'user/message', {role: 'user', content: 'next question'}),
    record(4, 'assistant/message', {message: {role: 'assistant', content: 'corrected answer'}}, {op: 'replace', startSeq: 2, endSeq: 2}),
  ]});
  assert.deepEqual(history.map(({content}) => content), ['question', 'corrected answer', 'next question']);
});

test('chat history hides runtime instructions and keeps the human conversation', () => {
  const history = sessionHistory({records: [
    record(1, 'system/message', {message: {role: 'system', content: 'internal policy'}}),
    record(2, 'user/message', {role: 'user', source: {kind: 'runtime-context'}, content: 'server details'}),
    record(3, 'user/message', {role: 'user', source: {kind: 'skill-catalog'}, content: 'skill instructions'}),
    record(4, 'user/message', {role: 'user', source: {kind: 'user'}, content: 'hello'}),
    record(5, 'assistant/message', {message: {role: 'assistant', content: 'ready'}}),
  ]});
  assert.deepEqual(history.map(message => message.content), ['hello', 'ready']);
});

test('native tool calls and their results stay in the conversation surface', () => {
  const call = {role: 'assistant', content: [{type: 'tool-call', id: 'call1', name: 'read', arguments: {path: 'config.toml'}}]};
  const result = {role: 'tool', callId: 'call1', content: [{type: 'text', text: 'model = "deepseek-v4.1-flash"'}]};
  const history = sessionHistory({records: [record(1, 'assistant/message', {message: call}), record(2, 'tool/result', {message: result})]});
  assert.deepEqual(history.map(({role, content}) => ({role, content})), [call, {role: result.role, content: result.content}]);
});
