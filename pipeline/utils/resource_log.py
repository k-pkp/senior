"""Append per-run resource usage to a shared log.csv (one row per run).

Captures: timestamp, command, input/output, device, GPU name, total time,
inference time, CPU count + load, RAM used + peak, VRAM used + peak.

Usage from orchestrator:
    from pipeline.utils.resource_log import prime_counters, append_run
    prime_counters(device)                       # at start
    ...
    append_run(log_path, args, device, total_time, inference_time)  # at end

File is created with a header if missing; subsequent runs append rows.
"""
import csv
import os
import platform
import sys
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

try:
    import resource as _resource
except ImportError:
    _resource = None


CSV_HEADER = [
    "timestamp", "command", "input", "output",
    "device", "gpu_name", "platform",
    "total_time_s", "inference_time_s",
    "cpu_count", "cpu_load_pct",
    "ram_used_gb", "ram_peak_gb",
    "vram_used_gb", "vram_peak_gb",
]


def prime_counters(device):
    """Call once at the start of a run to reset GPU peak counters and
    prime psutil's CPU sampler so the final reading is meaningful."""
    if psutil is not None:
        psutil.cpu_percent(interval=None)   # priming sample, ignored

    if torch is None:
        return
    dev = str(device)
    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _ram_used_gb():
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / 1e9


def _ram_peak_gb():
    if _resource is None:
        return _ram_used_gb()
    ru = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    # Linux: ru_maxrss in KB. macOS: in bytes.
    scale_bytes = ru if sys.platform == "darwin" else ru * 1024
    return scale_bytes / 1e9


def _gpu_info(device):
    """Return (name, used_gb, peak_gb)."""
    if torch is None:
        return "", float("nan"), float("nan")

    dev = str(device)
    if dev == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name()
        used = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        return name, used, peak

    if dev == "mps":
        used = float("nan")
        peak = float("nan")
        try:
            used = torch.mps.current_allocated_memory() / 1e9
        except Exception:
            pass
        try:
            peak = torch.mps.driver_allocated_memory() / 1e9
        except Exception:
            pass
        return "Apple MPS", used, peak

    return "", float("nan"), float("nan")


def _fmt(x, prec=3):
    if isinstance(x, float):
        if x != x:                          # NaN
            return ""
        return f"{x:.{prec}f}"
    return str(x)


def append_run(log_path, args, device, total_time, inference_time):
    """Append one row to log_path. Create file + header if missing."""
    gpu_name, vram_used, vram_peak = _gpu_info(device)

    cpu_count = psutil.cpu_count(logical=True) if psutil else os.cpu_count() or 0
    cpu_load = psutil.cpu_percent(interval=None) if psutil else float("nan")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "input": getattr(args, "image_folder", ""),
        "output": getattr(args, "output_dir", "") or "",
        "device": str(device),
        "gpu_name": gpu_name,
        "platform": f"{platform.system()} {platform.release()}",
        "total_time_s": _fmt(total_time, 2),
        "inference_time_s": _fmt(inference_time, 2),
        "cpu_count": cpu_count,
        "cpu_load_pct": _fmt(cpu_load, 1),
        "ram_used_gb": _fmt(_ram_used_gb()),
        "ram_peak_gb": _fmt(_ram_peak_gb()),
        "vram_used_gb": _fmt(vram_used),
        "vram_peak_gb": _fmt(vram_peak),
    }

    log_path = os.path.abspath(log_path)
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"  Logged run → {log_path}")
