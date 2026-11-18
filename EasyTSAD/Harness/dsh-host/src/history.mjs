/** Map the pinned SessionPage wire records to chat messages for thin clients. */
export function sessionHistory(page) {
  const records = page.records || page.events || page.entries || [];
  const surface = [];
  for (const record of records) {
    const event = record.type === 'event' ? record.event : record;
    if (!event || typeof event !== 'object') continue;
    const replacement = event.surfaceOp;
    let index = surface.length;
    if (replacement?.op === 'replace') {
      const first = surface.findIndex(item => item.seq >= replacement.startSeq && item.seq <= replacement.endSeq);
      if (first >= 0) index = first;
      for (let i = surface.length - 1; i >= 0; i--) {
        if (surface[i].seq >= replacement.startSeq && surface[i].seq <= replacement.endSeq) surface.splice(i, 1);
      }
    }
    const message = ['assistant/message', 'tool/result'].includes(event.type) ? event.data?.message
      : event.type === 'user/message' ? event.data
      : event.role ? event : null;
    if (!message || typeof message !== 'object') continue;
    const role = message.role || event.type.split('/')[0];
    if (role === 'system' || ['runtime-context', 'skill-catalog', 'system-prompt'].includes(message.source?.kind)) continue;
    surface.splice(index, 0, {...message, role, seq: event.seq, id: message.id || event.seq});
  }
  return surface;
}
