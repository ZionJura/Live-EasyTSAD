import {build} from 'esbuild';
import {createRequire} from 'node:module';
import {mkdir, cp, readFile, writeFile} from 'node:fs/promises';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
const root = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
await mkdir(join(root, 'dist'), {recursive:true});
await build({
  entryPoints: [join(root, 'src/main.ts')], outfile:join(root, 'dist/tui.mjs'),
  bundle:true, platform:'node', format:'esm', target:'node24', sourcemap:false,
  banner:{js:"import {createRequire as __createRequire} from 'node:module'; const require=__createRequire(import.meta.url);"},
  metafile:true,
});
// The renderer discovers its Markdown theme/assets relative to import.meta.url.
// Copy runtime resources, if present, alongside the standalone bundle.
const renderer = dirname(require.resolve('@earendil-works/pi-tui'));
for (const name of ['assets', 'native']) {
  try { await cp(join(renderer, name), join(root, 'dist', name), {recursive:true}); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}
// Notices travel with cached standalone builds, not just with repository sources.
await mkdir(join(root, 'dist/licenses'), {recursive:true});
await cp(join(root, 'vendor/deepseek-harness/LICENSE'), join(root, 'dist/licenses/deepseek-harness.txt'));
await cp(join(root, 'vendor/PROVENANCE.md'), join(root, 'dist/licenses/PROVENANCE.md'));
for (const dependency of ['@earendil-works/pi-tui', 'diff', 'saxes', 'marked', 'xmlchars', 'get-east-asian-width', 'smol-toml']) {
  let directory = dirname(require.resolve(dependency));
  while (true) {
    try {
      if (JSON.parse(await readFile(join(directory,'package.json'),'utf8')).name === dependency) break;
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const parent = dirname(directory);
    if (parent === directory) throw new Error(`Cannot locate dependency notice: ${dependency}`);
    directory = parent;
  }
  let copied = false;
  for (const name of ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'license']) {
    try {
      await cp(join(directory,name),join(root,'dist/licenses',`${dependency.replaceAll('/','-')}.txt`));
      copied = true;
      break;
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  if (!copied) {
    // These published tarballs omit their repository license; preserve it here.
    const fallback = {'@earendil-works/pi-tui':'pi-tui-0.80.7.txt',saxes:'saxes-6.0.0.txt'}[dependency];
    if (!fallback) throw new Error(`Missing dependency license: ${dependency}`);
    await cp(join(root,'vendor/licenses',fallback),join(root,'dist/licenses',`${dependency.replaceAll('/','-')}.txt`));
  }
}
await writeFile(join(root, 'dist/build-info.json'), JSON.stringify({
  renderer:'@earendil-works/pi-tui@0.80.7',
  upstream:'7248b5ec8f8769f882f12fd521504fa48e97bcf3',
  package:JSON.parse(await readFile(join(root, 'package.json'), 'utf8')).name,
}) + '\n');
