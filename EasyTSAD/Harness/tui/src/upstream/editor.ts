// Adapted from DeepSeek Harness 7248b5ec8f8769f882f12fd521504fa48e97bcf3.
// BSD-3-Clause; see ../../vendor/deepseek-harness/LICENSE and ../../vendor/PROVENANCE.md.
import { CURSOR_MARKER, Editor, truncateToWidth, visibleWidth } from '@earendil-works/pi-tui'
import { BRACKETED_PASTE_START, BRACKETED_PASTE_END, sanitizePastedText } from './text.js'

/** Editor that shows a placeholder without making it editable content. */
export class HintEditor extends Editor {
  private clipboardBuffer: string | undefined
  /** Placeholder shown in the empty input row; `undefined` hides it. */
  hint: string | undefined
  /** Prompt text rendered before the placeholder, matching the live prompt width. */
  hintPrefix = ''

  /** Keep terminal controls out of editable clipboard content, including split pastes. */
  override handleInput(data: string): void {
    if (this.clipboardBuffer !== undefined) {
      this.clipboardBuffer += data
      const end = this.clipboardBuffer.indexOf(BRACKETED_PASTE_END)
      if (end < 0) return
      const pasted = this.clipboardBuffer.slice(0, end)
      const remaining = this.clipboardBuffer.slice(end + BRACKETED_PASTE_END.length)
      this.clipboardBuffer = undefined
      super.handleInput(BRACKETED_PASTE_START + sanitizePastedText(pasted) + BRACKETED_PASTE_END)
      if (remaining) this.handleInput(remaining)
      return
    }
    const start = data.indexOf(BRACKETED_PASTE_START)
    if (start >= 0) {
      if (start > 0) super.handleInput(data.slice(0, start))
      this.clipboardBuffer = ''
      this.handleInput(data.slice(start + BRACKETED_PASTE_START.length))
      return
    }
    super.handleInput(data)
  }

  override render(width: number): string[] {
    const lines = super.render(width)
    if (this.hint === undefined || this.getText() !== '') return lines
    const content = lines[0]
    /* v8 ignore next -- Editor always renders one content row. */
    if (content === undefined) return lines
    const padding = ' '.repeat(this.getPaddingX())
    /* v8 ignore next -- the mounted editor is focused whenever its empty-input hint is rendered. */
    const marker = this.focused ? CURSOR_MARKER : ''
    const available = Math.max(0, width - visibleWidth(padding) - visibleWidth(this.hintPrefix))
    const placeholder = truncateToWidth(this.hint, available, '')
    const used = visibleWidth(padding) + visibleWidth(this.hintPrefix) + visibleWidth(placeholder)
    lines[0] = `${padding}${this.hintPrefix}${marker}${placeholder}${' '.repeat(Math.max(0, width - used))}`
    return lines
  }
}

