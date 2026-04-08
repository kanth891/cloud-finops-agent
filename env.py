"""
Cloud FinOps Cost Optimization — Core Environment.

Implements the OpenEnv-compliant interface:
  - reset()  → initial Observation
  - state()  → current Observation
  - step()   → (Observation, Reward, done, info)
  - get_final_result() → EpisodeResult
"""

import copy
import re
from typing import Dict, Tuple, Any, Optional, List

from models import (
    Observation, Action, Reward, CloudState, Server,
    InstanceSize, ServerEnvironment, EpisodeResult, INSTANCE_COST,
)
from tasks import (
    init_idle_killer, init_rightsizer, init_minefield,
)


class CloudFinOpsEnv:
    """
    Cloud FinOps Cost Optimization Environment.

    The agent observes a fleet of cloud servers and must reduce costs by
    terminating idle instances and right-sizing over-provisioned ones,
    while NEVER touching production servers.
    """

    # All valid instance sizes the agent can resize to
    VALID_SIZES: List[str] = [s.value for s in InstanceSize]

    def __init__(self, task_id: str = "idle_killer") -> None:
        """
        Initialise the environment.

        Args:
            task_id: One of "idle_killer", "rightsizer", or "minefield"
        """
        self.task_id: str = task_id
        self.task_config: Dict[str, Any] = self._load_task_config(task_id)

        self.servers: List[Server] = []
        self.initial_daily_cost: float = 0.0
        self.total_savings: float = 0.0
        self.step_count: int = 0
        self.max_steps: int = self.task_config["max_steps"]
        self.accumulated_rewards: List[float] = []
        self.production_violations: int = 0
        self.unnecessary_actions: int = 0
        self.grader: Any = self.task_config["grader"]
        self._done: bool = False

    # ------------------------------------------------------------------
    # Task loading
    # ------------------------------------------------------------------

    def _load_task_config(self, task_id: str) -> Dict[str, Any]:
        """Load task configuration by ID."""
        if task_id == "idle_killer":
            return init_idle_killer()
        elif task_id == "rightsizer":
            return init_rightsizer()
        elif task_id == "minefield":
            return init_minefield()
        else:
            raise ValueError(f"Unknown task: {task_id}")

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment to initial state and return first observation."""
        self.servers = copy.deepcopy(self.task_config["servers"])
        self.initial_daily_cost = sum(s.cost_per_day for s in self.servers)
        self.total_savings = 0.0
        self.step_count = 0
        self.accumulated_rewards = []
        self.production_violations = 0
        self.unnecessary_actions = 0
        self._done = False
        return self.state()

    def state(self) -> Observation:
        """Return the current observation."""
        cloud_state = CloudState(
            servers=self.servers,
            total_daily_cost=self._current_daily_cost(),
            total_savings=self.total_savings,
            step_count=self.step_count,
        )
        valid_actions = self._get_valid_actions()
        return Observation(cloud_state=cloud_state, valid_actions=valid_actions)

    def step(self, action_str: str) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Args:
            action_str: Action string, e.g.
                "terminate_instance(server_id='srv_01')"
                "resize_instance(server_id='srv_02', new_size='small')"
                "done()"

        Returns:
            (observation, reward, done, info)
        """
        self.step_count += 1
        reward_value: float = 0.0
        reward_components: Dict[str, float] = {}
        error_msg: Optional[str] = None

        try:
            if action_str.strip() == "done()":
                reward_value, reward_components = self._handle_done()
            elif action_str.startswith("terminate_instance"):
                reward_value, reward_components, error_msg = self._handle_terminate(action_str)
            elif action_str.startswith("resize_instance"):
                reward_value, reward_components, error_msg = self._handle_resize(action_str)
            else:
                error_msg = f"Unknown action: {action_str}"
                reward_value = -0.05
                reward_components = {"invalid_action": -0.05}
        except Exception as e:
            error_msg = str(e)
            reward_value = -0.05
            reward_components = {"error": -0.05}

        # Step penalty (encourages efficiency)
        step_penalty = -0.01
        reward_value += step_penalty
        reward_components["step_penalty"] = step_penalty

        self.accumulated_rewards.append(reward_value)

        # Check termination
        if not self._done:
            self._done = self._check_done()

        reward = Reward(
            value=reward_value,
            components=reward_components,
            info=error_msg or "Action executed successfully",
        )

        info: Dict[str, Any] = {
            "step": self.step_count,
            "total_savings": self.total_savings,
            "current_daily_cost": self._current_daily_cost(),
            "production_violations": self.production_violations,
            "error": error_msg,
        }

        return self.state(), reward, self._done, info

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_done(self) -> Tuple[float, Dict[str, float]]:
        """Agent signals it is finished optimising."""
        self._done = True
        # Small bonus for knowing when to stop
        bonus = 0.1 if self.total_savings > 0 else 0.0
        return bonus, {"done_bonus": bonus}

    def _handle_terminate(self, action_str: str) -> Tuple[float, Dict[str, float], Optional[str]]:
        """Parse and execute a terminate_instance action."""
        server_id = self._parse_server_id(action_str)
        if server_id is None:
            return -0.05, {"parse_error": -0.05}, "Could not parse server_id from action"

        server = self._find_server(server_id)
        if server is None:
            return -0.05, {"invalid_server": -0.05}, f"Server '{server_id}' not found"

        if server.is_terminated:
            self.unnecessary_actions += 1
            return -0.05, {"already_terminated": -0.05}, f"Server '{server_id}' is already terminated"

        # ——— PRODUCTION GUARD ———
        if server.environment == ServerEnvironment.PRODUCTION:
            self.production_violations += 1
            self._done = True
            return -1.0, {"production_violation": -1.0}, (
                f"CRITICAL: Attempted to terminate PRODUCTION server '{server_id}'! "
                "Episode terminated with penalty."
            )

        # Execute termination
        saved = server.cost_per_day
        server.is_terminated = True
        server.cost_per_day = 0.0
        server.cpu_usage_percent = 0.0
        server.memory_usage_percent = 0.0
        self.total_savings += saved

        reward = saved / 100.0  # Proportional to savings
        return reward, {"cost_savings": reward, "daily_saved": saved}, None

    def _handle_resize(self, action_str: str) -> Tuple[float, Dict[str, float], Optional[str]]:
        """Parse and execute a resize_instance action."""
        server_id, new_size_str = self._parse_resize_args(action_str)
        if server_id is None or new_size_str is None:
            return -0.05, {"parse_error": -0.05}, "Could not parse resize arguments"

        server = self._find_server(server_id)
        if server is None:
            return -0.05, {"invalid_server": -0.05}, f"Server '{server_id}' not found"

        if server.is_terminated:
            self.unnecessary_actions += 1
            return -0.05, {"already_terminated": -0.05}, f"Server '{server_id}' is terminated"

        # Validate new size
        try:
            new_size = InstanceSize(new_size_str)
        except ValueError:
            return -0.05, {"invalid_size": -0.05}, (
                f"Invalid size '{new_size_str}'. Must be one of: {self.VALID_SIZES}"
            )

        if new_size == server.instance_type:
            self.unnecessary_actions += 1
            return -0.05, {"same_size": -0.05}, f"Server '{server_id}' is already '{new_size_str}'"

        # ——— PRODUCTION GUARD ———
        if server.environment == ServerEnvironment.PRODUCTION:
            self.production_violations += 1
            self._done = True
            return -1.0, {"production_violation": -1.0}, (
                f"CRITICAL: Attempted to resize PRODUCTION server '{server_id}'! "
                "Episode terminated with penalty."
            )

        # Execute resize
        old_cost = server.cost_per_day
        new_cost = INSTANCE_COST[new_size]
        saved = old_cost - new_cost  # Can be negative if upsizing

        server.instance_type = new_size
        server.cost_per_day = new_cost

        if saved > 0:
            self.total_savings += saved

        reward = saved / 100.0  # Proportional — negative if upsizing
        components: Dict[str, float] = {"cost_savings": reward, "daily_saved": saved}
        if saved < 0:
            components["upsize_penalty"] = saved / 100.0

        return reward, components, None

    # ------------------------------------------------------------------
    # Valid action generation
    # ------------------------------------------------------------------

    def _get_valid_actions(self) -> List[str]:
        """Generate all currently valid action strings."""
        actions: List[str] = ["done()"]

        for server in self.servers:
            if server.is_terminated:
                continue
            if server.environment == ServerEnvironment.PRODUCTION:
                continue  # Don't offer production actions (hint to the agent)

            # Terminate is always valid for non-prod, non-terminated
            actions.append(f"terminate_instance(server_id='{server.server_id}')")

            # Resize to each different size
            for size in InstanceSize:
                if size != server.instance_type:
                    actions.append(
                        f"resize_instance(server_id='{server.server_id}', new_size='{size.value}')"
                    )

        return actions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_server(self, server_id: str) -> Optional[Server]:
        """Find a server by ID."""
        return next((s for s in self.servers if s.server_id == server_id), None)

    def _current_daily_cost(self) -> float:
        """Sum of costs for all active (non-terminated) servers."""
        return sum(s.cost_per_day for s in self.servers if not s.is_terminated)

    def _check_done(self) -> bool:
        """Check if the episode should end."""
        if self._done:
            return True
        if self.step_count >= self.max_steps:
            return True
        return False

    @staticmethod
    def _parse_server_id(action_str: str) -> Optional[str]:
        """Extract server_id from an action string."""
        match = re.search(r"server_id=['\"]([^'\"]+)['\"]", action_str)
        return match.group(1) if match else None

    @staticmethod
    def _parse_resize_args(action_str: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract server_id and new_size from a resize action string."""
        sid_match = re.search(r"server_id=['\"]([^'\"]+)['\"]", action_str)
        size_match = re.search(r"new_size=['\"]([^'\"]+)['\"]", action_str)
        server_id = sid_match.group(1) if sid_match else None
        new_size = size_match.group(1) if size_match else None
        return server_id, new_size

    # ------------------------------------------------------------------
    # Final result (for grading)
    # ------------------------------------------------------------------

    def get_final_result(self) -> EpisodeResult:
        """Compute and return the final graded episode result."""
        cloud_state = CloudState(
            servers=self.servers,
            total_daily_cost=self._current_daily_cost(),
            total_savings=self.total_savings,
            step_count=self.step_count,
        )

        # Build a preliminary result for the grader
        preliminary = EpisodeResult(
            success=self.production_violations == 0,
            steps=self.step_count,
            score=0.0,
            rewards=self.accumulated_rewards,
            total_savings=self.total_savings,
            initial_daily_cost=self.initial_daily_cost,
            production_violations=self.production_violations,
            unnecessary_actions=self.unnecessary_actions,
        )

        # Let the task grader compute the final score
        score = self.grader.grade(cloud_state, preliminary)

        return EpisodeResult(
            success=self.production_violations == 0,
            steps=self.step_count,
            score=score,
            rewards=self.accumulated_rewards,
            total_savings=self.total_savings,
            initial_daily_cost=self.initial_daily_cost,
            production_violations=self.production_violations,
            unnecessary_actions=self.unnecessary_actions,
        )
