/** Launch the pinned published DSH runtime for chat or isolated CLI workers. */
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { buildPatch } from './src/profile.mjs';

const hostDir = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const argv = process.argv.slice(2);
function start(executable, args, env, cwd) {
  const child = spawn(executable, args, {cwd, env, stdio: 'inherit'});
  child.on('error', error => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
  for (const event of ['SIGINT', 'SIGTERM']) process.on(event, () => child.kill(event));
  child.on('exit', (code, reason) => { process.exitCode = code ?? (reason ? 1 : 0); });
}
const pinnedNode = join(hostDir, 'node_modules/node/bin/node');
if (Number(process.versions.node.split('.')[0]) < 24) {
  if (!existsSync(pinnedNode)) throw new Error('Node >=24.9 is required. Run npm ci in EasyTSAD/Harness/dsh-host.');
  start(pinnedNode, [fileURLToPath(import.meta.url), ...argv], process.env);
} else {
  const env = {...process.env};
  const root = resolve(env.TSAD_ROOT || join(hostDir, '../../..'));
  const home = resolve(env.DSH_HOME || join(root, 'runtime/harness-workbench/dsh'));
  env.DSH_HOME = home; env.TSAD_ROOT = root; env.DSH_TELEMETRY = '0';
  const packagePath = require.resolve('@deepseek-ai/dsh/package.json');
  const pkg = JSON.parse(readFileSync(packagePath, 'utf8'));
  if (pkg.version !== '0.2.2-rc.1') throw new Error(`DSH version mismatch: ${pkg.version}`);
  const bin = resolve(dirname(packagePath), typeof pkg.bin === 'string' ? pkg.bin : pkg.bin.dsh);
  if (argv[0] === '--worker') {
    // Preserve native headless JSON/session semantics. Python does not implement Typert.
    start(process.execPath, [bin, ...argv.slice(1)], env, root);
  } else {
    const profileName = 'tsad-interactive', profileDir = join(home, 'profiles', profileName);
    mkdirSync(profileDir, {recursive: true});
    const manifest = join(profileDir, 'package.json');
    // This profile is launcher-owned. Migrate existing Web-backed profiles as well.
    const profile = existsSync(manifest) ? JSON.parse(readFileSync(manifest, 'utf8')) : {name: 'dsh-profile-tsad-interactive', private: true, dependencies: {}};
    profile.dsh = {...profile.dsh, profile: {...profile.dsh?.profile, bundles: ['@deepseek-ai/dsh-base']}};
    writeFileSync(manifest, JSON.stringify(profile, null, 2) + '\n');
    const userPatch = join(profileDir, 'cordis.patch.yml');
    if (!existsSync(userPatch)) writeFileSync(userPatch, '[]\n');
    const patchPath = join(home, 'tsad-interactive.generated.patch.yml');
    writeFileSync(patchPath, JSON.stringify(buildPatch(env, hostDir), null, 2) + '\n');
    const args = [bin, '--profile', profileName, '--patch', patchPath];
    if (argv.includes('--check')) args.push('--dump-config');
    start(process.execPath, args, env, root);
  }
}
