import assert from 'node:assert/strict'
import test from 'node:test'
import type { Terminal } from '@earendil-works/pi-tui'
import { HarnessView } from '../src/view.js'
import { extractDiffs } from '../src/upstream/tools.js'

class MemoryTerminal implements Terminal {
  columns = 100
  rows = 80
  kittyProtocolActive = false
  input: (value: string) => void = () => {}
  resize: () => void = () => {}
  output = ''
  start(input: (value: string) => void, resize: () => void): void { this.input = input; this.resize = resize }
  stop(): void {}
  async drainInput(): Promise<void> {}
  write(value: string): void { this.output += value }
  moveBy(): void {}
  hideCursor(): void {}
  showCursor(): void {}
  clearLine(): void {}
  clearFromCursor(): void {}
  clearScreen(): void {}
  setTitle(): void {}
  setProgress(): void {}
}
const plain = (value: string): string => value.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '').replace(/\x1b_pi:c\x07/g, '')
const tick = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 30))

test('original tool cards pair nested results and render real file diffs, shell, search and TSAD output', async () => {
  const terminal = new MemoryTerminal()
  const view = new HarnessView({terminal, onSubmit() {}, onCancel() {}, onQuit() {}})
  view.setDetails(true)
  view.setMessages([
    {role: 'user', content: '修复评测并说明原因'},
    {role: 'assistant', content: [{type: 'reasoning', text: 'Check evaluation'}, {type: 'text', text: '**Checking**'}, {type: 'tool-call', id: 'edit1', name: 'fs-edit', arguments: '{"path":"score.py","oldText":"old\\n","newText":"new\\n"}'}]},
    {role: 'tool', content: [{type: 'tool-result', toolCallId: 'edit1', content: [{type: 'text', text: 'applied'}]}]},
    {role: 'assistant', content: [{type: 'tool-call', id: 'shell1', name: 'bash', arguments: '{"command":"pytest","cwd":"/workspace"}'}]},
    {role: 'tool', toolCallId: 'shell1', content: '{"stdout":"3 passed","exitCode":0}'},
    {role: 'assistant', content: [{type: 'tool-call', id: 'find1', name: 'web_search', arguments: '{"query":"TSAD evaluation"}'}]},
    {role: 'tool', toolCallId: 'find1', content: 'Official paper'},
    {role: 'assistant', content: [{type: 'tool-call', id: 'job1', name: 'tsad_status', arguments: '{}'}]},
    {role: 'tool', toolCallId: 'job1', content: '{"stage":"evaluating","score":0.91}'},
    {role: 'assistant', content: [{type: 'tool-call', id: 'meta1', name: 'fs-write', arguments: '{"path":"meta.py"}'}]},
    {role: 'tool', toolCallId: 'meta1', content: 'saved', meta: {diffs: [{path: 'meta.py', oldText: null, newText: 'metadata_diff'}]}},
  ])
  view.start()
  try {
    await tick()
    const screen = plain(terminal.output)
    assert.match(screen, /Tool \/ fs-edit/)
    assert.equal((screen.match(/Tool \/ fs-edit/g) || []).length, 1)
    assert.match(screen, /\+ new/); assert.match(screen, /- old/)
    assert.match(screen, /3 passed/); assert.match(screen, /exit 0/)
    assert.match(screen, /Official paper/); assert.match(screen, /evaluating/)
    assert.match(screen, /Check evaluation/)
    assert.match(screen, /\+ metadata_diff/)
    assert.doesNotMatch(screen, /Ctrl\+O/)
  } finally { view.stop() }
})

test('four basic operations work and a fragmented malicious paste cannot emit terminal commands', async () => {
  const terminal = new MemoryTerminal()
  const sent: string[] = []
  let cancelled = 0, quit = 0
  const view = new HarnessView({terminal, onSubmit(text) {sent.push(text)}, onCancel() {cancelled++}, onQuit() {quit++}})
  view.start()
  try {
    terminal.input('first')
    terminal.input('\x1b[13;2u')
    terminal.input('second')
    terminal.input('\r')
    assert.deepEqual(sent, ['first\nsecond'])
    terminal.input('\x1b[200~safe\x1b]52;c;')
    terminal.input('c2VjcmV0\x07\x1b[2Jtext\x1b[201~')
    terminal.input('\r')
    assert.equal(sent[1], 'safetext')
    terminal.input('\x0b'); terminal.input('\x11')
    assert.equal(cancelled, 1); assert.equal(quit, 1)
    terminal.input('/model'); terminal.input('\t')
    await tick()
    assert.doesNotMatch(terminal.output, /\x1b\]52;c;/)
    assert.doesNotMatch(plain(terminal.output), /Select model|Resume session/)
  } finally { view.stop() }
})

test('model/tool output escapes OSC, C0/C1 and CSI before Markdown rendering', async () => {
  const terminal = new MemoryTerminal()
  const view = new HarnessView({terminal, onSubmit() {}, onCancel() {}, onQuit() {}})
  view.setMessages([{role: 'assistant', content: 'unsafe\x1b]52;c;payload\x07\x1b[2J\x9dtitle'}])
  view.start()
  try {
    await tick()
    assert.doesNotMatch(terminal.output, /\x1b\]52;c;payload|\x9dtitle/)
    assert.match(plain(terminal.output), /\\x1b\]52;c;payload/)
  } finally { view.stop() }
  assert.deepEqual(extractDiffs({path: 'a.py', diffs: [{oldText: null, newText: 'new'}]}), [{path: 'a.py', oldText: null, newText: 'new'}])
  assert.deepEqual(extractDiffs({path: 'a.py', content: 'unknown previous state'}), [])
})

test('history refresh and narrow terminal resize preserve the focused input', async () => {
  const terminal = new MemoryTerminal()
  const sent: string[] = []
  const view = new HarnessView({terminal, onSubmit(text) { sent.push(text) }, onCancel() {}, onQuit() {}})
  view.start()
  try {
    terminal.input('正在输入')
    const history = Array.from({length: 80}, (_, index) => ({role: index % 2 ? 'assistant' : 'user', content: `第 ${index} 条消息 中文内容 **Markdown**`}))
    view.setMessages(history)
    terminal.columns = 24; terminal.rows = 12; terminal.resize()
    view.setStatus('1 agent · evaluating')
    await tick()
    terminal.input('\x7f'); terminal.input('中'); terminal.input('\r')
    assert.deepEqual(sent, ['正在输中'])
    assert.match(plain(terminal.output), /Markdown/)
  } finally { view.stop() }
})
