"""
Cloud FinOps Cost Optimization — Pydantic Data Models.

Defines all typed Observation, Action, and Reward models
for the OpenEnv-compliant environment.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InstanceSize(str, Enum):
    """Available cloud instance sizes, ordered smallest to largest."""
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


class ServerEnvironment(str, Enum):
    """Deployment environments — production servers must NEVER be touched."""
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


# ---------------------------------------------------------------------------
# Cost mapping
# ---------------------------------------------------------------------------

INSTANCE_COST: Dict[InstanceSize, float] = {
    InstanceSize.NANO: 5.0,
    InstanceSize.SMALL: 20.0,
    InstanceSize.MEDIUM: 50.0,
    InstanceSize.LARGE: 100.0,
    InstanceSize.XLARGE: 200.0,
}


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class Server(BaseModel):
    """A single cloud server instance."""
    server_id: str = Field(..., description="Unique server identifier (e.g. srv_01)")
    instance_type: InstanceSize = Field(..., description="Current instance size")
    cost_per_day: float = Field(..., description="Daily cost in USD")
    cpu_usage_percent: float = Field(..., description="Current CPU utilization 0-100")
    memory_usage_percent: float = Field(..., description="Current memory utilization 0-100")
    environment: ServerEnvironment = Field(..., description="Deployment environment")
    is_terminated: bool = Field(default=False, description="Whether server has been shut down")


class CloudState(BaseModel):
    """Snapshot of the entire cloud infrastructure."""
    servers: List[Server] = Field(default_factory=list, description="All servers in the fleet")
    total_daily_cost: float = Field(default=0.0, description="Sum of all active server costs")
    total_savings: float = Field(default=0.0, description="Cumulative savings achieved so far")
    step_count: int = Field(default=0, description="Number of actions taken so far")


# ---------------------------------------------------------------------------
# OpenEnv interface models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees after each step."""
    cloud_state: CloudState
    valid_actions: List[str] = Field(
        default_factory=list,
        description="List of action strings the agent can execute right now",
    )

    class Config:
        arbitrary_types_allowed = True


class Action(BaseModel):
    """An action the agent wants to perform."""
    action_type: str = Field(
        ...,
        description="One of: terminate_instance, resize_instance, done",
    )
    server_id: Optional[str] = Field(
        default=None,
        description="Target server (required for terminate/resize)",
    )
    new_size: Optional[str] = Field(
        default=None,
        description="Target instance size (required for resize)",
    )


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = Field(..., description="Scalar reward value")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of reward into named components",
    )
    info: str = Field(default="", description="Human-readable explanation")


class EpisodeResult(BaseModel):
    """Final result after an episode completes — used by graders."""
    success: bool = Field(..., description="True if no production violations occurred")
    steps: int = Field(..., description="Total actions taken")
    score: float = Field(..., description="Final graded score 0.0-1.0")
    rewards: List[float] = Field(default_factory=list, description="Per-step rewards")
    total_savings: float = Field(default=0.0, description="Total daily $ saved")
    initial_daily_cost: float = Field(default=0.0, description="Starting daily cost")
    production_violations: int = Field(default=0, description="Number of production touches")
    unnecessary_actions: int = Field(default=0, description="Actions with zero impact")
