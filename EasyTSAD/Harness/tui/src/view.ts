/** Slim original DSH pi-tui presentation backed exclusively by the remote host. */
import { Container, Key, ProcessTerminal, Spacer, Text, TUI, matchesKey, visibleWidth, type Terminal } from '@earendil-works/pi-tui'
import { HintEditor } from './upstream/editor.js'
import { createPalette, markdownTheme, selectTheme } from './upstream/theme.js'
import { displayInlineText, displayText } from './upstream/text.js'
import { contentText, parseArguments } from './upstream/content.js'
import { AssistantMessageComponent, ToolCardComponent, UserMessageComponent } from './upstream/transcript.js'
import type { ContentBlock, FileDiff, ToolDefinition } from './upstream/types.js'
import { toolPresenter } from './upstream/tools.js'

export interface HarnessMessage {
  role: string
  content: string | readonly unknown[]
  id?: string
  name?: string
  toolName?: string
  toolCallId?: string
  isError?: boolean
  meta?: unknown
  /** Structured file diffs, if supplied by the host; never inferred as edits. */
  diffs?: FileDiff[]
}
export interface HarnessViewCallbacks {
  onSubmit(text: string): void | Promise<void>
  onCancel(): void | Promise<void>
  onQuit(): void | Promise<void>
  /** Optional terminal injection for offline renderer/PTY tests. */
  terminal?: Terminal
}

/** Normalize both DSH content blocks and OpenAI-style text blocks for display. */
export function normalizeContent(content: HarnessMessage['content']): ContentBlock[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }]
  const result: ContentBlock[] = []
  for (const value of content) {
    if (!value || typeof value !== 'object') continue
    const block = value as Record<string, unknown>
    const kind = block.type
    if (kind === 'tool-call' || kind === 'tool_call') {
      const args = block.arguments ?? block.input ?? {}
      result.push({ type: 'tool-call', id: String(block.id ?? ''), name: String(block.name ?? 'tool'), arguments: typeof args === 'string' ? args : JSON.stringify(args) })
    } else if (kind === 'tool-result' || kind === 'tool_result') {
      const nested = block.content
      result.push({ type: 'tool-result', content: normalizeContent(typeof nested === 'string' || Array.isArray(nested) ? nested : JSON.stringify(nested ?? '')), isError: block.isError === true })
    } else if (typeof block.text === 'string') {
      result.push({ type: kind === 'reasoning' || kind === 'thinking' ? 'reasoning' : 'text', text: block.text })
    } else if (typeof block.thinking === 'string') {
      result.push({ type: 'reasoning', text: block.thinking })
    } else {
      result.push({ type: 'text', text: `[${String(kind ?? 'content')}]` })
    }
  }
  return result
}

export class HarnessView {
  private readonly palette = createPalette(!process.env.NO_COLOR)
  private readonly theme = markdownTheme(this.palette)
  private readonly ui: TUI
  private readonly chat = new Container()
  private readonly status = new Text('', 1, 0)
  private readonly notice = new Text('', 1, 0)
  private readonly editor: HintEditor
  private messages: readonly HarnessMessage[] = []
  private details = false
  private started = false

  constructor(private readonly callbacks: HarnessViewCallbacks) {
    this.ui = new TUI(callbacks.terminal ?? new ProcessTerminal(), true)
    const prompt = `${this.palette.bold(this.palette.accent('tsad'))} ${this.palette.dim('> ' )}`
    this.editor = new HintEditor(this.ui, { borderColor: this.palette.dim, selectList: selectTheme(this.palette) }, {
      paddingX: 1, frame: 'none', prompt: { first: prompt, continuation: ' '.repeat(visibleWidth(prompt)) },
    })
    this.editor.hintPrefix = prompt
    this.editor.hint = 'Enter send · Shift+Enter newline · Ctrl+K cancel · Ctrl+Q quit'
    this.editor.onSubmit = (text: string) => {
      if (!text.trim()) return
      this.editor.setText('')
      this.invoke(() => callbacks.onSubmit(text))
    }
    this.ui.addChild(new Text(`${this.palette.bold(this.palette.accent('DEEPSEEK'))} ${this.palette.bold('HARNESS')}  ${this.palette.dim('TSAD')}`, 1, 0))
    this.ui.addChild(new Spacer(1))
    this.ui.addChild(this.chat)
    this.ui.addChild(new Spacer(1))
    this.ui.addChild(this.notice)
    this.ui.addChild(this.status)
    this.ui.addChild(this.editor)
    this.ui.setFocus(this.editor)
    this.ui.addInputListener((data) => {
      if (matchesKey(data, Key.ctrl('k'))) {
        this.invoke(() => callbacks.onCancel()); return { consume: true }
      }
      if (matchesKey(data, Key.ctrl('q'))) {
        this.invoke(() => callbacks.onQuit()); return { consume: true }
      }
      // Menus, model/resume selectors and alternate control actions are absent.
      return undefined
    })
  }

  private invoke(action: () => void | Promise<void>): void {
    try { Promise.resolve(action()).catch(error => this.setNotice(String(error))) }
    catch (error) { this.setNotice(String(error)) }
  }
  start(): void {
    if (this.started) return
    try { this.ui.start(); this.started = true }
    catch (error) { this.ui.stop(); throw error }
  }
  stop(): void { if (this.started) { this.ui.stop(); this.started = false } }
  setStatus(text: string): void { this.status.setText(this.palette.dim(displayText(text))); this.ui.requestRender() }
  setNotice(text: string): void { this.notice.setText(this.palette.warning(displayText(text))); this.ui.requestRender() }
  setDetails(enabled: boolean): void { if (this.details !== enabled) { this.details = enabled; this.rebuild() } }
  setMessages(messages: readonly HarnessMessage[]): void { this.messages = messages; this.rebuild() }

  private rebuild(): void {
    this.chat.clear()
    const calls = new Map<string, ToolCardComponent>()
    for (const message of this.messages) {
      const content = normalizeContent(message.content)
      if (message.role === 'tool' || message.role === 'tool-result') {
        const wrapped = Array.isArray(message.content) ? message.content.find(item => !!item && typeof item === 'object' && ['tool-result', 'tool_result'].includes(String((item as Record<string, unknown>).type))) as Record<string, unknown> | undefined : undefined
        const callId = message.toolCallId ?? String(wrapped?.toolCallId ?? wrapped?.callId ?? wrapped?.id ?? '')
        let card = callId ? calls.get(callId) : undefined
        if (!card) { card = this.card(message.name ?? message.toolName ?? String(wrapped?.name ?? 'tool'), '{}', message.diffs); this.chat.addChild(card) }
        const nested = content.length === 1 && content[0]?.type === 'tool-result' ? content[0].content : content
        card.updateResult({ message: { content: [{ content: nested, isError: message.isError || wrapped?.isError === true }] }, meta: message.meta ?? wrapped?.meta })
      } else if (message.role === 'assistant') {
        this.chat.addChild(new AssistantMessageComponent(content, this.details, this.palette, this.theme))
        for (const block of content) {
          if (block.type !== 'tool-call') continue
          const card = this.card(block.name, block.arguments, message.diffs)
          calls.set(block.id, card); this.chat.addChild(card)
        }
      } else {
        this.chat.addChild(new Spacer(1))
        this.chat.addChild(new UserMessageComponent(contentText(content), this.palette, this.theme, message.role === 'user' ? 'You' : displayInlineText(message.role)))
      }
    }
    this.ui.requestRender()
  }

  private card(name: string, args: string, diffs?: FileDiff[]): ToolCardComponent {
    name = displayInlineText(name)
    const parsed = parseArguments(args)
    const definition = toolPresenter(name, parsed.value, diffs)
    const card = new ToolCardComponent(name, parsed, definition, 6, 1000, this.palette, this.theme)
    card.setVisibility(this.details ? 'expanded' : 'collapsed')
    return card
  }
}
