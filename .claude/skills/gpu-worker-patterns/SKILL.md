---
name: gpu-worker-patterns
description: Patterns for GPU-intensive Python workers. Use when building ML inference services, 3D rendering pipelines, or batch processing systems requiring GPU acceleration. Covers factory patterns, lazy model loading, CUDA memory management, error handling, and background task processing.
---

# GPU Worker Patterns

Patterns for building reliable GPU-intensive Python services.

## Service Layer Architecture

### Factory Pattern for Mock/Real Switching

```python
from config.settings import get_settings

def create_service(real_class, mock_class):
    """Factory to switch between mock and real implementations."""
    settings = get_settings()
    return mock_class() if settings.MOCK_MODE else real_class()
```

### Base Service Interface

```python
from abc import ABC, abstractmethod

class BaseService(ABC):
    @abstractmethod
    async def execute(self, *args, **kwargs) -> dict:
        """Must return dict with 'success' bool and 'mock' flag."""
        pass
```

## Lazy Model Loading

Load models only on first use to reduce startup time:

```python
class GPUService:
    def __init__(self):
        self._model = None

    def _ensure_model_loaded(self):
        if self._model is None:
            import torch
            self._model = load_model().to(self.device)
            self._model.eval()

    async def execute(self, input_data):
        self._ensure_model_loaded()
        # Use model...
```

## CUDA Memory Management

```python
import torch

# Always use no_grad for inference
with torch.no_grad():
    result = model(input_tensor)

# Clear cache after large operations
torch.cuda.empty_cache()

# Check memory before large allocations
if torch.cuda.memory_allocated() > threshold:
    torch.cuda.empty_cache()
```

## Error Handling

```python
from fastapi import HTTPException

try:
    result = await gpu_operation()
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    raise HTTPException(503, "GPU memory exhausted, try again")
except Exception as e:
    logger.error(f"GPU operation failed: {e}")
    raise HTTPException(500, str(e))
```

## Background Tasks for Long Operations

```python
from fastapi import BackgroundTasks

@router.post("/long-operation")
async def trigger(request: Request, bg_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    bg_tasks.add_task(long_running_gpu_task, job_id, request.data)
    return {"status": "queued", "job_id": job_id}
```

## Temp File Management

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp_dir:
    # Process files in tmp_dir
    output_path = Path(tmp_dir) / "result.glb"
    # Auto-cleanup on exit
```

## Response Structure

Always return consistent response format:

```python
return {
    "success": True,
    "result": processed_data,
    "processing_time_ms": elapsed_ms,
    "mock": settings.MOCK_MODE,
}
```

## Health Check Pattern

```python
@router.get("/health")
async def health():
    settings = get_settings()
    response = {"status": "healthy", "mock_mode": settings.MOCK_MODE}

    if not settings.MOCK_MODE:
        import torch
        response["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            response["gpu_name"] = torch.cuda.get_device_name(0)

    return response
```
