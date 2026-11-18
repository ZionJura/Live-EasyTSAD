"""Trusted experiment events. One writer per run directory, readable during training."""
import datetime as dt
import json
import os


def append_event(run_dir, identity, event, curve=None, **data):
    row = dict(version=1, at=dt.datetime.now(dt.timezone.utc).isoformat(),
               type=event, **identity, data=data)
    if curve is not None:
        row['curve'] = curve
    payload = (json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode()
    fd = os.open(str(run_dir / 'progress.jsonl'), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        if os.write(fd, payload) != len(payload):
            raise OSError('short progress event write')
    finally:
        os.close(fd)
    return row
