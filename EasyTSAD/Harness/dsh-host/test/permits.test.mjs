import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,mkdir,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {acquirePermit} from '../src/model-permits.mjs';

test('shared slots wait, release and recover crashed owners', async()=>{
  const root=await mkdtemp(join(tmpdir(),'tsad-permits-'));
  try {
    const release=await acquirePermit(root,1);
    const stop=new AbortController();
    const queued=acquirePermit(root,1,stop.signal);
    stop.abort(new Error('cancelled'));
    await assert.rejects(queued,/cancelled/);
    await release();
    await mkdir(join(root,'0'));
    await writeFile(join(root,'0/owner.json'),JSON.stringify({pid:2147483647,identity:'dead',leaseId:'old'}));
    const recovered=await acquirePermit(root,1);
    await recovered();
    await recovered();
  } finally {await rm(root,{recursive:true,force:true});}
});
