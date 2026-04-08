"""
Cloud FinOps Cost Optimization — Task Definitions & Graders.

Defines three escalating tasks and their deterministic graders:
  1. idle_killer   (easy)   — terminate idle dev servers
  2. rightsizer    (medium) — downsize over-provisioned instances
  3. minefield     (hard)   — mixed optimization with production guards
"""

from typing import Dict, Any, List
from models import (
    CloudState, Server, InstanceSize, ServerEnvironment,
    EpisodeResult, INSTANCE_COST,
)


# ═══════════════════════════════════════════════════════════════════════════
# Grader base class
# ═══════════════════════════════════════════════════════════════════════════

class TaskGrader:
    """Base class for task graders.  Every grader returns a float in [0, 1]."""

    def grade(self, final_state: CloudState, episode_result: EpisodeResult) -> float:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Task 1 — The Idle Killer (Easy)
# ═══════════════════════════════════════════════════════════════════════════

class IdleKillerGrader(TaskGrader):
    """
    Score = savings_achieved / max_possible_savings
    Bonus +0.1 if zero production violations (capped at 1.0).
    """

    def grade(self, final_state: CloudState, episode_result: EpisodeResult) -> float:
        # Max possible savings: both idle dev servers terminated
        # srv_02: $50/day, srv_04: $20/day  →  $70/day max
        max_savings = 70.0
        savings_ratio = min(1.0, episode_result.total_savings / max_savings) if max_savings > 0 else 0.0
        score = savings_ratio
        if episode_result.production_violations == 0:
            score = min(1.0, score + 0.1)
        return max(0.0, min(1.0, score))


def _build_idle_killer_servers() -> List[Server]:
    """5 servers — 2 are idle dev boxes, 3 are healthy."""
    return [
        Server(
            server_id="srv_01", instance_type=InstanceSize.LARGE,
            cost_per_day=INSTANCE_COST[InstanceSize.LARGE],
            cpu_usage_percent=72.0, memory_usage_percent=65.0,
            environment=ServerEnvironment.PRODUCTION,
        ),
        Server(
            server_id="srv_02", instance_type=InstanceSize.MEDIUM,
            cost_per_day=INSTANCE_COST[InstanceSize.MEDIUM],
            cpu_usage_percent=0.0, memory_usage_percent=2.0,
            environment=ServerEnvironment.DEV,
        ),
        Server(
            server_id="srv_03", instance_type=InstanceSize.SMALL,
            cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
            cpu_usage_percent=45.0, memory_usage_percent=38.0,
            environment=ServerEnvironment.STAGING,
        ),
        Server(
            server_id="srv_04", instance_type=InstanceSize.SMALL,
            cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
            cpu_usage_percent=0.0, memory_usage_percent=1.0,
            environment=ServerEnvironment.DEV,
        ),
        Server(
            server_id="srv_05", instance_type=InstanceSize.NANO,
            cost_per_day=INSTANCE_COST[InstanceSize.NANO],
            cpu_usage_percent=88.0, memory_usage_percent=70.0,
            environment=ServerEnvironment.PRODUCTION,
        ),
    ]


def init_idle_killer() -> Dict[str, Any]:
    return {
        "servers": _build_idle_killer_servers(),
        "grader": IdleKillerGrader(),
        "max_steps": 50,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task 2 — The Rightsizer (Medium)
# ═══════════════════════════════════════════════════════════════════════════

class RightsizerGrader(TaskGrader):
    """
    Score = savings_weight * savings_ratio + efficiency_weight * efficiency
    savings_weight   = 0.7
    efficiency_weight = 0.3  (fewer steps = higher efficiency)
    """

    def grade(self, final_state: CloudState, episode_result: EpisodeResult) -> float:
        # Max possible savings:
        # srv_02 xlarge→small: $200-$20 = $180
        # srv_05 large→small:  $100-$20 = $80
        # srv_08 large→small:  $100-$20 = $80
        # Total: $340/day
        max_savings = 340.0
        savings_ratio = min(1.0, episode_result.total_savings / max_savings) if max_savings > 0 else 0.0

        # Efficiency: ideal is 3 resize actions + 1 done = 4 steps
        ideal_steps = 4
        efficiency = max(0.0, 1.0 - (episode_result.steps - ideal_steps) / 20.0)

        score = 0.7 * savings_ratio + 0.3 * efficiency
        if episode_result.production_violations > 0:
            score *= 0.3  # Heavy penalty
        return max(0.0, min(1.0, score))


def _build_rightsizer_servers() -> List[Server]:
    """10 servers — 3 are massively over-provisioned, rest are fine."""
    return [
        # Well-sized servers
        Server(
            server_id="srv_01", instance_type=InstanceSize.MEDIUM,
            cost_per_day=INSTANCE_COST[InstanceSize.MEDIUM],
            cpu_usage_percent=60.0, memory_usage_percent=55.0,
            environment=ServerEnvironment.PRODUCTION,
        ),
        # Over-provisioned #1 — xlarge at 5% CPU
        Server(
            server_id="srv_02", instance_type=InstanceSize.XLARGE,
            cost_per_day=INSTANCE_COST[InstanceSize.XLARGE],
            cpu_usage_percent=5.0, memory_usage_percent=8.0,
            environment=ServerEnvironment.STAGING,
        ),
        Server(
            server_id="srv_03", instance_type=InstanceSize.SMALL,
            cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
            cpu_usage_percent=78.0, memory_usage_percent=65.0,
            environment=ServerEnvironment.PRODUCTION,
        ),
        Server(
            server_id="srv_04", instance_type=InstanceSize.MEDIUM,
            cost_per_day=INSTANCE_COST[InstanceSize.MEDIUM],
            cpu_usage_percent=55.0, memory_usage_percent=48.0,
            environment=ServerEnvironment.DEV,
        ),
        # Over-provisioned #2 — large at 3% CPU
        Server(
            server_id="srv_05", instance_type=InstanceSize.LARGE,
            cost_per_day=INSTANCE_COST[InstanceSize.LARGE],
            cpu_usage_percent=3.0, memory_usage_percent=5.0,
            environment=ServerEnvironment.STAGING,
        ),
        Server(
            server_id="srv_06", instance_type=InstanceSize.NANO,
            cost_per_day=INSTANCE_COST[InstanceSize.NANO],
            cpu_usage_percent=90.0, memory_usage_percent=85.0,
            environment=ServerEnvironment.PRODUCTION,
        ),
        Server(
            server_id="srv_07", instance_type=InstanceSize.SMALL,
            cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
            cpu_usage_percent=42.0, memory_usage_percent=35.0,
            environment=ServerEnvironment.DEV,
        ),
        # Over-provisioned #3 — large at 7% CPU
        Server(
            server_id="srv_08", instance_type=InstanceSize.LARGE,
            cost_per_day=INSTANCE_COST[InstanceSize.LARGE],
            cpu_usage_percent=7.0, memory_usage_percent=10.0,
            environment=ServerEnvironment.DEV,
        ),
        Server(
            server_id="srv_09", instance_type=InstanceSize.MEDIUM,
            cost_per_day=INSTANCE_COST[InstanceSize.MEDIUM],
            cpu_usage_percent=65.0, memory_usage_percent=58.0,
            environment=ServerEnvironment.PRODUCTION,
        ),
        Server(
            server_id="srv_10", instance_type=InstanceSize.SMALL,
            cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
            cpu_usage_percent=50.0, memory_usage_percent=40.0,
            environment=ServerEnvironment.STAGING,
        ),
    ]


def init_rightsizer() -> Dict[str, Any]:
    return {
        "servers": _build_rightsizer_servers(),
        "grader": RightsizerGrader(),
        "max_steps": 50,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task 3 — The Minefield (Hard)
# ═══════════════════════════════════════════════════════════════════════════

class MinefieldGrader(TaskGrader):
    """
    Score = 0.50 * savings_ratio
          + 0.20 * efficiency
          + 0.30 * production_safety

    Production safety = 1.0 if zero violations, else 0.0.
    """

    def grade(self, final_state: CloudState, episode_result: EpisodeResult) -> float:
        # Max savings:
        # Terminate: srv_03($20), srv_06($5), srv_11($50), srv_15($5) = $80
        # Resize: srv_05 xlarge→small($180), srv_09 large→small($80), srv_14 xlarge→small($180) = $440
        # Total: $520
        max_savings = 520.0
        savings_ratio = min(1.0, episode_result.total_savings / max_savings) if max_savings > 0 else 0.0

        # Efficiency: ideal 7 targeted actions + 1 done = 8 steps
        ideal_steps = 8
        efficiency = max(0.0, 1.0 - (episode_result.steps - ideal_steps) / 30.0)

        # Production safety: binary — all or nothing
        production_safety = 1.0 if episode_result.production_violations == 0 else 0.0

        score = 0.50 * savings_ratio + 0.20 * efficiency + 0.30 * production_safety
        return max(0.0, min(1.0, score))


def _build_minefield_servers() -> List[Server]:
    """
    20 servers — a realistic mixed fleet:
      - 8 production servers (must NOT be touched)
      - 4 idle dev servers   (should be terminated)
      - 3 oversized staging  (should be resized)
      - 5 healthy non-prod   (should be left alone)
    """
    servers = [
        # --- Production servers (DO NOT TOUCH) ---
        Server(server_id="srv_01", instance_type=InstanceSize.LARGE,
               cost_per_day=INSTANCE_COST[InstanceSize.LARGE],
               cpu_usage_percent=78.0, memory_usage_percent=72.0,
               environment=ServerEnvironment.PRODUCTION),
        Server(server_id="srv_02", instance_type=InstanceSize.XLARGE,
               cost_per_day=INSTANCE_COST[InstanceSize.XLARGE],
               cpu_usage_percent=85.0, memory_usage_percent=80.0,
               environment=ServerEnvironment.PRODUCTION),
        # Idle dev #1 — should terminate
        Server(server_id="srv_03", instance_type=InstanceSize.SMALL,
               cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
               cpu_usage_percent=0.0, memory_usage_percent=1.0,
               environment=ServerEnvironment.DEV),
        Server(server_id="srv_04", instance_type=InstanceSize.MEDIUM,
               cost_per_day=INSTANCE_COST[InstanceSize.MEDIUM],
               cpu_usage_percent=62.0, memory_usage_percent=55.0,
               environment=ServerEnvironment.PRODUCTION),
        # Oversized staging #1 — should resize
        Server(server_id="srv_05", instance_type=InstanceSize.XLARGE,
               cost_per_day=INSTANCE_COST[InstanceSize.XLARGE],
               cpu_usage_percent=4.0, memory_usage_percent=6.0,
               environment=ServerEnvironment.STAGING),
        # Idle dev #2 — should terminate
        Server(server_id="srv_06", instance_type=InstanceSize.NANO,
               cost_per_day=INSTANCE_COST[InstanceSize.NANO],
               cpu_usage_percent=0.0, memory_usage_percent=0.5,
               environment=ServerEnvironment.DEV),
        Server(server_id="srv_07", instance_type=InstanceSize.LARGE,
               cost_per_day=INSTANCE_COST[InstanceSize.LARGE],
               cpu_usage_percent=70.0, memory_usage_percent=68.0,
               environment=ServerEnvironment.PRODUCTION),
        Server(server_id="srv_08", instance_type=InstanceSize.MEDIUM,
               cost_per_day=INSTANCE_COST[InstanceSize.MEDIUM],
               cpu_usage_percent=55.0, memory_usage_percent=50.0,
               environment=ServerEnvironment.PRODUCTION),
        # Oversized staging #2 — should resize
        Server(server_id="srv_09", instance_type=InstanceSize.LARGE,
               cost_per_day=INSTANCE_COST[InstanceSize.LARGE],
               cpu_usage_percent=6.0, memory_usage_percent=9.0,
               environment=ServerEnvironment.STAGING),
        # Healthy dev (leave alone)
        Server(server_id="srv_10", instance_type=InstanceSize.SMALL,
               cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
               cpu_usage_percent=55.0, memory_usage_percent=48.0,
               environment=ServerEnvironment.DEV),
        # Idle dev #3 — should terminate
        Server(server_id="srv_11", instance_type=InstanceSize.MEDIUM,
               cost_per_day=INSTANCE_COST[InstanceSize.MEDIUM],
               cpu_usage_percent=0.0, memory_usage_percent=2.0,
               environment=ServerEnvironment.DEV),
        Server(server_id="srv_12", instance_type=InstanceSize.SMALL,
               cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
               cpu_usage_percent=88.0, memory_usage_percent=75.0,
               environment=ServerEnvironment.PRODUCTION),
        # Healthy staging (leave alone)
        Server(server_id="srv_13", instance_type=InstanceSize.SMALL,
               cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
               cpu_usage_percent=42.0, memory_usage_percent=38.0,
               environment=ServerEnvironment.STAGING),
        # Oversized staging #3 — should resize
        Server(server_id="srv_14", instance_type=InstanceSize.XLARGE,
               cost_per_day=INSTANCE_COST[InstanceSize.XLARGE],
               cpu_usage_percent=3.0, memory_usage_percent=5.0,
               environment=ServerEnvironment.STAGING),
        # Idle dev #4 — should terminate
        Server(server_id="srv_15", instance_type=InstanceSize.NANO,
               cost_per_day=INSTANCE_COST[InstanceSize.NANO],
               cpu_usage_percent=0.0, memory_usage_percent=0.0,
               environment=ServerEnvironment.DEV),
        Server(server_id="srv_16", instance_type=InstanceSize.MEDIUM,
               cost_per_day=INSTANCE_COST[InstanceSize.MEDIUM],
               cpu_usage_percent=68.0, memory_usage_percent=60.0,
               environment=ServerEnvironment.PRODUCTION),
        # Healthy dev (leave alone)
        Server(server_id="srv_17", instance_type=InstanceSize.NANO,
               cost_per_day=INSTANCE_COST[InstanceSize.NANO],
               cpu_usage_percent=40.0, memory_usage_percent=30.0,
               environment=ServerEnvironment.DEV),
        # Healthy staging (leave alone)
        Server(server_id="srv_18", instance_type=InstanceSize.MEDIUM,
               cost_per_day=INSTANCE_COST[InstanceSize.MEDIUM],
               cpu_usage_percent=50.0, memory_usage_percent=45.0,
               environment=ServerEnvironment.STAGING),
        Server(server_id="srv_19", instance_type=InstanceSize.LARGE,
               cost_per_day=INSTANCE_COST[InstanceSize.LARGE],
               cpu_usage_percent=75.0, memory_usage_percent=70.0,
               environment=ServerEnvironment.PRODUCTION),
        # Healthy dev (leave alone)
        Server(server_id="srv_20", instance_type=InstanceSize.SMALL,
               cost_per_day=INSTANCE_COST[InstanceSize.SMALL],
               cpu_usage_percent=60.0, memory_usage_percent=52.0,
               environment=ServerEnvironment.DEV),
    ]
    return servers


def init_minefield() -> Dict[str, Any]:
    return {
        "servers": _build_minefield_servers(),
        "grader": MinefieldGrader(),
        "max_steps": 50,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Module-level grader functions (referenced by openenv.yaml)
# ═══════════════════════════════════════════════════════════════════════════

def grade_idle_killer(final_state: CloudState, episode_result: EpisodeResult) -> float:
    return IdleKillerGrader().grade(final_state, episode_result)


def grade_rightsizer(final_state: CloudState, episode_result: EpisodeResult) -> float:
    return RightsizerGrader().grade(final_state, episode_result)


def grade_minefield(final_state: CloudState, episode_result: EpisodeResult) -> float:
    return MinefieldGrader().grade(final_state, episode_result)
