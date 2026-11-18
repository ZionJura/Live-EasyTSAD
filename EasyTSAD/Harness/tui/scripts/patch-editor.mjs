// Apply the editor changes shipped by the fixed original DSH TUI source.
// npm ci is idempotent; refuse unknown or partially patched dependency versions.
import {spawnSync} from 'node:child_process';
import {readFileSync, existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const patch = fileURLToPath(new URL('../patches/@earendil-works__pi-tui@0.80.7.patch', import.meta.url));
if (!existsSync(patch)) throw new Error('Missing pinned official pi-tui editor patch');
const pkg = JSON.parse(readFileSync(new URL('../node_modules/@earendil-works/pi-tui/package.json', import.meta.url), 'utf8'));
if (pkg.version !== '0.80.7') throw new Error(`Unexpected pi-tui version ${pkg.version}`);
const directory = `${root}node_modules/@earendil-works/pi-tui`;
const args = ['--batch', '--silent', '-p1', '-d', directory, '-i', patch];
const applied = spawnSync('patch', [...args, '--forward', '--dry-run'], {encoding:'utf8'});
if (applied.status === 0) {
  const result = spawnSync('patch', [...args, '--forward'], {encoding:'utf8'});
  if (result.status !== 0) throw new Error(`Cannot patch pi-tui: ${result.stderr || result.stdout}`);
} else if (spawnSync('patch', [...args, '--reverse', '--dry-run'], {encoding:'utf8'}).status !== 0) {
  throw new Error(`pi-tui does not match the pinned original patch: ${applied.stderr || applied.stdout}`);
}
