/** A read-only human slash-command plugin. No model or experiment is started. */
export const name = 'tsad-guide-command'
export const inject = ['commands']
export function apply(ctx) {
  ctx.commands.register({
    name: 'hello-tsad',
    description: '显示当前 Agent 会话，不运行实验',
    handler: ({ agent }) => ({ kind: 'success', text: `TSAD guide: session ${agent.id}` }),
  })
}
