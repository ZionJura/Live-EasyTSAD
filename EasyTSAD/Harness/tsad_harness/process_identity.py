"""Portable identity of a local process; remote PIDs must never be passed here."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def process_identity(pid: int) -> str | None:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if sys.platform == "darwin":
        # libproc supplies the creation timestamp at microsecond resolution.
        try:
            class BSDInfo(ctypes.Structure):
                _fields_ = [("flags", ctypes.c_uint32), ("status", ctypes.c_uint32),
                            ("rest", ctypes.c_uint32 * 10), ("comm", ctypes.c_char * 16),
                            ("name", ctypes.c_char * 32), ("extra", ctypes.c_uint32 * 6),
                            ("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64)]
            lib = ctypes.CDLL("/usr/lib/libproc.dylib")
            info = BSDInfo()
            if lib.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info) or info.status == 5:
                return None
            return f"darwin:{info.start_sec}:{info.start_usec}"
        except (OSError, ValueError, OverflowError):
            return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, IndexError):
        return None


if __name__ == "__main__":
    try:
        value = process_identity(int(sys.argv[1]))
    except (IndexError, ValueError):
        value = None
    if value:
        print(value)
    else:
        raise SystemExit(1)
