---
name: mock-mode-patterns
description: Patterns for mock/real mode switching in backend services. Use when building services that need to work without expensive dependencies (GPUs, external APIs) during development and testing. Covers configuration, service factories, mock data guidelines, and testing strategies.
---

# Mock Mode Patterns

Patterns for seamless mock/real service switching.

## Configuration Pattern

Use environment variable with safe default (mock mode ON by default):

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    MOCK_MODE: bool = True  # Safe default

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

## Service Factory Pattern

```python
from abc import ABC, abstractmethod

class BaseService(ABC):
    @abstractmethod
    async def execute(self, *args, **kwargs) -> dict:
        pass

def create_service(real_class, mock_class):
    from config.settings import get_settings
    return mock_class() if get_settings().MOCK_MODE else real_class()
```

## Mock Implementation Guidelines

### Match Schema Exactly

Mock responses must have identical structure to real responses:

```python
class MockService(BaseService):
    async def execute(self, input_data):
        return {
            "success": True,
            "result": mock_result,
            "processing_time_ms": 100.0,
            "mock": True,  # Always include this flag
        }
```

### Realistic Timing

Simulate realistic delays:

```python
import asyncio

async def execute(self, data):
    await asyncio.sleep(0.1)  # Simulate processing
    return {"success": True, "mock": True}
```

### Use Fixture Files

Store mock assets in `assets/mock/`:

```
assets/mock/
├── avatar_template.glb
├── sample_video.mp4
└── sample_thumbnail.jpg
```

### Derive Mock Data from Input

Make mock data correlate with input for realistic testing:

```python
async def execute(self, height_cm: float):
    # Mock measurements scale with height
    return {
        "success": True,
        "measurements": {
            "chest_cm": height_cm * 0.52,
            "waist_cm": height_cm * 0.44,
        },
        "mock": True,
    }
```

## Service Facade Pattern

```python
class AvatarService:
    def __init__(self):
        self._impl = create_service(RealGenerator, MockGenerator)

    async def generate(self, *args, **kwargs):
        return await self._impl.execute(*args, **kwargs)
```

## Testing Both Modes

```python
import pytest

@pytest.fixture(params=[True, False], ids=["mock", "real"])
def mock_mode(request, monkeypatch):
    monkeypatch.setenv("MOCK_MODE", str(request.param).lower())
    get_settings.cache_clear()
    return request.param

def test_service(mock_mode):
    service = create_service(RealService, MockService)
    result = service.execute(data)
    assert result["mock"] == mock_mode
```

## API Response Pattern

Include mock flag in all API responses:

```python
class ResponseModel(BaseModel):
    success: bool
    data: Optional[Dict]
    processing_time_ms: float
    mock: bool  # Always include

@router.post("/endpoint")
async def endpoint(request: Request) -> ResponseModel:
    result = await service.execute(request)
    return ResponseModel(**result)
```

## Quick Check Helper

```python
def is_mock_mode() -> bool:
    from config.settings import get_settings
    return get_settings().MOCK_MODE
```

## Logging Pattern

```python
from loguru import logger

async def execute(self, data):
    logger.info(f"[MOCK] Processing request")
    # ... mock implementation
```

## Database Operations in Mock Mode

Return mock data without hitting database:

```python
async def fetch_user(user_id: str):
    if is_mock_mode():
        return {"id": user_id, "name": "Mock User"}
    return await db.fetch_one(query)
```
