#!/usr/bin/env node
/** Display first; all service discovery and Agent work happens asynchronously. */
import {parseArgs} from 'node:util';
import {HarnessView} from './view.js';
import {RemoteHarnessAdapter} from './adapter.js';

const help = `Usage: scripts/harness tui [task] [--resume sessionId|latest] [--gpus auto|0,1] [--refresh seconds]

Early DeepSeek terminal UI, connected to the existing TSAD background service.
Enter send · Shift+Enter newline · Ctrl+K stop reply · Ctrl+Q detach
/help commands · /details [on|off] tool output · /new [--background] · /sessions list · /agent <id> switch · /connect reconnect
`;
let options: ReturnType<typeof parseArgs>;
try {
  options = parseArgs({options:{resume:{type:'string'}, gpus:{type:'string'}, refresh:{type:'string',default:'2'}, task:{type:'string'}, help:{type:'boolean',short:'h'}}, allowPositionals:true});
} catch (error) { process.stderr.write(`${String(error)}\n${help}`); process.exit(2); }
const values = options.values;
if (values.help) { process.stdout.write(help); process.exit(0); }
const refresh = Number(values.refresh);
if (!Number.isFinite(refresh) || refresh < .1 || (options.positionals?.length ?? 0) > 1) {
  process.stderr.write(`Invalid arguments. --refresh must be at least 0.1 seconds.\n${help}`); process.exit(2);
}
if (!process.stdin.isTTY || !process.stdout.isTTY) {
  process.stderr.write('Harness TUI requires an interactive terminal. Use scripts/harness --help for CLI commands.\n'); process.exit(2);
}

let adapter: RemoteHarnessAdapter;
let details = false, stopping = false;
const quit = () => {
  if (stopping) return;
  stopping = true;
  adapter?.dispose();
  view.stop();
  // A detached background service owns its sessions, workers and experiments.
  process.exit(0);
};
const view = new HarnessView({
  onSubmit: async raw => {
    const line = raw.trim();
    if (/^\/details(?:\s|$)/.test(line)) {
      const value = line.slice('/details'.length).trim();
      if (value && value !== 'on' && value !== 'off') throw new Error('Usage: /details [on|off]');
      details = value ? value === 'on' : !details;
      view.setDetails(details);
      view.setNotice(`工具输出：${details ? '展开' : '折叠'}`);
    } else if (line.startsWith('/')) await adapter.command(line);
    else await adapter.send(raw);
  },
  onCancel: () => adapter.cancel(),
  onQuit: quit,
});
view.setStatus('正在连接后台… Ctrl+Q 退出');
view.setNotice('默认新建独立会话；/sessions 查看后台会话，/new --background 异步创建，/agent <id> 切换。');
view.start();
adapter = new RemoteHarnessAdapter({
  root:process.env.TSAD_ROOT, refresh,
  task: typeof values.task === 'string' ? values.task : options.positionals?.[0],
  resume:typeof values.resume === 'string' ? values.resume : undefined,
  gpus:typeof values.gpus === 'string' ? values.gpus : undefined,
  onMessages: messages => view.setMessages(messages),
  onStatus: status => view.setStatus(status),
  onNotice: notice => view.setNotice(notice),
});
process.once('SIGTERM', quit);
process.once('SIGINT', quit);
process.once('SIGHUP', quit);
process.once('uncaughtException', error => {
  adapter.dispose(); view.stop(); process.stderr.write(`${error.stack || error.message}\n`); process.exit(1);
});
// Yield the event loop so the first frame precedes configuration and HTTP reads.
setImmediate(() => { void adapter.connect().catch(error => {
  if (!stopping) view.setNotice(`连接失败：${String(error)}；输入 /connect 重试`);
}); });
