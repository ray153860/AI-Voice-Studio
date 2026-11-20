from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class EngineRequest:
    input_audio: str | None = None
    text: str | None = None
    reference_audio: str | None = None
    model_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineResult:
    output_audio: str
    sample_rate: int
    engine: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    name: str
    capability: str

    async def run(self, request: EngineRequest) -> EngineResult: ...

class PlaceholderEngine:
    def __init__(self, capability: str):
        self.name, self.capability = "placeholder", capability
    async def run(self, **kwargs):
        raise RuntimeError(f"{self.capability} engine is an interface stub; no AI inference is installed")
