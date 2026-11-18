/** Remote tool presenters. These format DTOs and never execute tools. */
import { contentText } from './content.js'
import type { ContentBlock, FileDiff, ToolDefinition, ToolResultView } from './types.js'

const record = (value: unknown): Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const valueText = (value: unknown): string => typeof value === 'string' ? value : ''

/** Accept explicit old/new text pairs only; do not invent file contents. */
export function extractDiffs(value: unknown, fallbackPath = ''): FileDiff[] {
  if (Array.isArray(value)) return value.flatMap(item => extractDiffs(item, fallbackPath))
  const item = record(value)
  const path = valueText(item.path ?? item.filePath ?? item.file_path ?? item.file) || fallbackPath
  const before = item.oldText ?? item.old_text ?? item.oldString ?? item.old_string ?? item.before
  const after = item.newText ?? item.new_text ?? item.newString ?? item.new_string ?? item.after
  const originalNull = item.oldText === null || item.old_text === null || item.before === null
  if (path && typeof after === 'string' && (typeof before === 'string' || originalNull)) {
    return [{ path, oldText: originalNull ? null : before as string, newText: after }]
  }
  for (const key of ['diffs', 'edits', 'changes', 'files', 'result', 'data']) {
    if (item[key] !== undefined) {
      const diffs = extractDiffs(item[key], path)
      if (diffs.length) return diffs
    }
  }
  return []
}

function resultJson(content: ContentBlock[]): unknown {
  try { return JSON.parse(contentText(content)) } catch { return undefined }
}

export function toolPresenter(name: string, args: unknown, suppliedDiffs?: FileDiff[]): ToolDefinition | undefined {
  const input = record(args)
  const path = valueText(input.path ?? input.filePath ?? input.file_path ?? input.file)
  const slug = name.toLowerCase().replace(/[_/]/g, '-')
  const diffs = suppliedDiffs?.length ? suppliedDiffs : extractDiffs(args)
  const terminal = /^(bash|shell|exec|exec-command|execute|terminal)$/.test(slug)
  const read = /^(read|read-file|fs-read|file-read)$/.test(slug)
  const search = /search|grep|glob/.test(slug)
  const web = /fetch|browse|crawl/.test(slug)
  const tsad = slug.startsWith('tsad-')
  const edit = /^(write|write-file|fs-write|edit|edit-file|fs-edit|apply-patch)$/.test(slug)
  if (!(terminal || read || search || web || tsad || edit || diffs.length)) return undefined
  const title = terminal ? valueText(input.command ?? input.cmd) || name
    : read ? `Read ${path || name}`
    : search ? `Search ${valueText(input.query ?? input.pattern ?? input.q) || path || name}`
    : web ? `Fetch ${valueText(input.url) || name}`
    : path ? `${name} ${path}` : name
  return {
    presentCall: () => diffs.length ? { card: 'diff', title, diffs }
      : terminal ? { card: 'terminal', title, description: valueText(input.description), cwd: valueText(input.cwd) }
      : { card: 'generic', title, rawInput: args },
    presentResult: (_args, result): ToolResultView => {
      const raw = resultJson(result.content)
      const metaDiffs = extractDiffs(result.meta, path)
      const resultDiffs = metaDiffs.length ? metaDiffs : extractDiffs(raw, path)
      const allDiffs = resultDiffs.length ? resultDiffs : diffs
      if (allDiffs.length) return { card: 'diff', title, diffs: allDiffs }
      if (terminal) {
        const output = record(raw)
        return { card: 'terminal', title, output: valueText(output.output ?? output.stdout) || contentText(result.content),
          ...(typeof output.exitCode === 'number' ? { exitCode: output.exitCode } : {}),
          ...(typeof output.signal === 'string' ? { signal: output.signal } : {}) }
      }
      return { card: read ? 'read' : 'generic', title, content: result.content }
    },
  }
}
