/** Client display types only. No DSH/Cordis agent runtime is loaded here. */
export type JsonValue = unknown
export type ContentBlock =
  | { type: 'text' | 'reasoning'; text: string }
  | { type: 'tool-call'; id: string; name: string; arguments: string }
  | { type: 'tool-result'; content: ContentBlock[]; isError?: boolean }
export interface TodoItem { content: string; status: string }
export interface FileDiff { path: string; oldText: string | null; newText: string }
type GenericView = { card: 'generic' | 'read'; title: string; content?: ContentBlock[]; rawInput?: unknown }
type DiffView = { card: 'diff'; title: string; diffs: FileDiff[] }
export interface TerminalCallView { card: 'terminal'; title: string; description?: string; cwd?: string; output?: string; exitCode?: number; signal?: string }
type OtherView = { card: 'search' | 'web'; title: string }
export type ToolCallView = GenericView | DiffView | TerminalCallView | OtherView
export type ToolResultView = ToolCallView
export interface ToolDefinition {
  presentCall?(args: unknown): ToolCallView | undefined
  presentResult?(args: unknown, result: { content: ContentBlock[]; isError: boolean; meta?: JsonValue }): ToolResultView | undefined
}
