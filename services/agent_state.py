from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AgentStage(str, Enum):
    INIT = "init"
    ROUTE = "route"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    VALIDATE = "validate"
    PERSIST = "persist"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentEvent:
    stage: str
    name: str
    data: dict = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AgentState:
    user_id: str
    session_id: str
    query: str
    current_stage: AgentStage = AgentStage.INIT
    events: list[AgentEvent] = field(default_factory=list)

    def transition(self, stage: AgentStage, name: str, data: dict | None = None):
        self.current_stage = stage
        self.events.append(
            AgentEvent(
                stage=stage.value,
                name=name,
                data=data or {},
            )
        )

    def fail(self, name: str, error: Exception | str):
        self.transition(
            AgentStage.ERROR,
            name,
            {"error": str(error), "error_type": type(error).__name__},
        )

    def to_dict(self, max_events: int | None = None) -> dict:
        events = self.events if max_events is None else self.events[-max_events:]
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "current_stage": self.current_stage.value,
            "events": [
                {
                    "stage": event.stage,
                    "name": event.name,
                    "data": event.data,
                    "ts": event.ts,
                }
                for event in events
            ],
        }

