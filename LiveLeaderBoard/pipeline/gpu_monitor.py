"""Record CUDA allocator peaks and sampled process memory for one experiment."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path


class GPUMonitor:
    def __init__(self, run_dir: Path, interval: float = 5.0):
        self.run_dir = Path(run_dir)
        self.interval = interval
        self.pid = os.getpid()
        self.started = time.perf_counter()
        self.phase = "setup"
        self.phase_started = self.started
        self.phase_seconds = {}
        self.process_peaks = {}
        self.samples = 0
        self.errors = []
        self.torch = None
        self.devices = []
        self.finished = False
        self._stop = threading.Event()
        self._thread = None
        self._csv_file = None

    def _error(self, error):
        message = f"{type(error).__name__}: {error}"
        if message not in self.errors and len(self.errors) < 8:
            self.errors.append(message)

    def start(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    torch.cuda.reset_peak_memory_stats(index)
                    self.devices.append({"logical_index": index, "name": props.name,
                                         "total_memory_mib": props.total_memory / 2**20})
        except Exception as error:
            self._error(error)
        if self.devices:
            self._csv_file = (self.run_dir / "gpu_memory.csv").open("w", newline="")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(["elapsed_s", "phase", "gpu_uuid", "process_used_mib"])
            self._sample()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        else:
            self._write_report()
        return self

    def set_phase(self, phase):
        now = time.perf_counter()
        self.phase_seconds[self.phase] = self.phase_seconds.get(self.phase, 0.0) + now - self.phase_started
        self.phase_started = now
        self.phase = phase

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 3 or parts[0] != str(self.pid):
                    continue
                used = float(parts[2])
                gpu_uuid = parts[1]
                self.process_peaks[gpu_uuid] = max(self.process_peaks.get(gpu_uuid, 0.0), used)
                self.samples += 1
                self._csv.writerow([round(time.perf_counter() - self.started, 3),
                                    self.phase, gpu_uuid, used])
            self._csv_file.flush()
        except Exception as error:
            self._error(error)
        self._write_report()

    def _write_report(self):
        devices = []
        for device in self.devices:
            record = dict(device)
            try:
                index = device["logical_index"]
                record["peak_allocated_mib"] = self.torch.cuda.max_memory_allocated(index) / 2**20
                record["peak_reserved_mib"] = self.torch.cuda.max_memory_reserved(index) / 2**20
            except Exception as error:
                self._error(error)
            devices.append(record)
        phase_seconds = dict(self.phase_seconds)
        if not self.finished:
            phase_seconds[self.phase] = phase_seconds.get(self.phase, 0.0) + time.perf_counter() - self.phase_started
        report = {
            "schema_version": 1, "pid": self.pid,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": bool(self.devices), "finished": self.finished,
            "elapsed_s": round(time.perf_counter() - self.started, 3),
            "phase": self.phase,
            "phase_seconds": {key: round(value, 3) for key, value in phase_seconds.items()},
            "torch_devices": devices,
            "nvidia_smi_sample_interval_s": self.interval,
            "nvidia_smi_samples": self.samples,
            "nvidia_smi_peak_process_used_mib": dict(self.process_peaks),
            "errors": list(self.errors),
        }
        path = self.run_dir / "gpu_memory.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(tmp, path)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.devices:
            for device in self.devices:
                try:
                    self.torch.cuda.synchronize(device["logical_index"])
                except Exception as error:
                    self._error(error)
            self._sample()
        self.set_phase("finished")
        self.finished = True
        self._write_report()
        if self._csv_file is not None:
            self._csv_file.close()
