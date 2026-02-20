"""
Shared diagnostics for structured fallback logging across all ML pipeline stages.
"""

import logging
import time
from dataclasses import dataclass, field


@dataclass
class LogEntry:
    stage: str
    error_type: str
    message: str
    gpu_info: str
    timestamp: float = field(default_factory=time.time)
    duration_s: float = 0.0


class PipelineLog:
    """Collects structured log entries across all pipeline stages."""

    def __init__(self):
        self.entries: list[LogEntry] = []

    def append(self, entry: LogEntry):
        self.entries.append(entry)

    def to_dicts(self) -> list[dict]:
        return [
            {
                "stage": e.stage,
                "error_type": e.error_type,
                "message": e.message,
                "gpu_info": e.gpu_info,
                "duration_s": round(e.duration_s, 2),
            }
            for e in self.entries
        ]

    def __bool__(self):
        return len(self.entries) > 0

    def __iter__(self):
        return iter(self.entries)


def get_gpu_snapshot() -> str:
    """Return VRAM used/total as a string, or empty if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_mem / 1e9
            return f"GPU: {allocated:.1f}/{total:.1f} GB"
    except Exception:
        pass
    return ""


def log_fallback(
    logger: logging.Logger,
    stage: str,
    error: Exception,
    pipeline_log: PipelineLog | None = None,
    duration_s: float = 0.0,
):
    """Log a fallback event with GPU state and optional structured collection."""
    gpu_info = get_gpu_snapshot()
    gpu_suffix = f" | {gpu_info}" if gpu_info else ""

    logger.warning(
        "[%s] Fallback: %s: %s%s",
        stage,
        type(error).__name__,
        error,
        gpu_suffix,
    )

    if pipeline_log is not None:
        pipeline_log.append(
            LogEntry(
                stage=stage,
                error_type=type(error).__name__,
                message=str(error),
                gpu_info=gpu_info,
                duration_s=duration_s,
            )
        )
