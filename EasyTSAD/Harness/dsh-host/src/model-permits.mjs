/** Shared with Python CLI workers. mkdir and owner identity serialize model permits. */
import { mkdir, readFile, writeFile, stat, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
const execFileAsync = promisify(execFile);
export async function identity(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  if (process.platform === 'darwin') {
    try {
      const script = fileURLToPath(new URL('../../tsad_harness/process_identity.py', import.meta.url));
      const { stdout } = await execFileAsync(process.env.TSAD_HARNESS_PYTHON || '/usr/bin/python3', [script, String(pid)], { timeout: 5000 });
      return stdout.trim() || null;
    } catch { return null; }
  }
  try { return (await readFile(`/proc/${pid}/stat`, 'utf8')).split(') ').at(-1).split(' ')[19]; }
  catch { return null; }
}
async function ownerAt(path) {
  try { return JSON.parse(await readFile(join(path,'owner.json'),'utf8')); }
  catch { return null; }
}
async function expired(path) {
  const owner=await ownerAt(path);
  if (owner) return (await identity(owner.pid))!==owner.identity;
  return Date.now()-(await stat(path).catch(()=>({mtimeMs:Date.now()}))).mtimeMs>30000;
}
async function lock(path,owner,signal) {
  while (true) {
    signal?.throwIfAborted();
    try { await mkdir(path); await writeFile(join(path,'owner.json'),JSON.stringify(owner)); return; }
    catch (error) {
      if (error.code!=='EEXIST') throw error;
      if (await expired(path)) await rm(path,{recursive:true,force:true});
      else await delay(50,undefined,{signal});
    }
  }
}
export async function acquirePermit(directory,limit,signal) {
  await mkdir(directory,{recursive:true});
  const owner={pid:process.pid,identity:await identity(process.pid),startTime:Date.now()/1000,leaseId:randomUUID()};
  while (true) {
    signal?.throwIfAborted();
    for (let index=0;index<limit;index++) {
      const path=join(directory,String(index)), mutex=path+'.reclaim';
      await lock(mutex,owner,signal);
      let acquired=false;
      try {
        if (await expired(path)) await rm(path,{recursive:true,force:true});
        try { await mkdir(path); await writeFile(join(path,'owner.json'),JSON.stringify(owner)); acquired=true; }
        catch (error) { if (error.code!=='EEXIST') throw error; }
      } finally { await rm(mutex,{recursive:true,force:true}); }
      if (acquired) {
        let released=false;
        return async () => {
          if (released) return; released=true;
          await lock(mutex,owner);
          try { if ((await ownerAt(path))?.leaseId===owner.leaseId) await rm(path,{recursive:true,force:true}); }
          finally { await rm(mutex,{recursive:true,force:true}); }
        };
      }
    }
    await delay(100,undefined,{signal});
  }
}
