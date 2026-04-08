"""
Cloud FinOps Cost Optimization — Baseline Inference Script.

Uses an LLM (via OpenAI-compatible API) to evaluate the agent on all
three tasks.  Reads API credentials from environment variables:
  - HF_TOKEN     (or API_KEY)
  - API_BASE_URL
  - MODEL_NAME
"""

import os
import sys
import re
import json
from typing import Optional, List

from env import CloudFinOpsEnv
from models import Observation


# ──────────────────────────────────────────────────────────────────────────
# Try importing the OpenAI client
# ──────────────────────────────────────────────────────────────────────────

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore
    print("[WARN] openai package not installed. Install with: pip install openai", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────

class FinOpsAgent:
    """LLM-powered Cloud FinOps optimisation agent."""

    SYSTEM_PROMPT = (
        "You are a Cloud FinOps cost-optimization expert. Your job is to "
        "reduce cloud spending by terminating idle servers and right-sizing "
        "over-provisioned instances.  NEVER touch production servers.\n\n"
        "Rules:\n"
        "1. Terminate servers with 0% CPU in dev/staging environments.\n"
        "2. Resize servers with very low CPU (<10%) to a smaller instance type.\n"
        "3. NEVER terminate or resize a production server.\n"
        "4. When you are done optimizing, call done().\n\n"
        "Respond with ONLY the action command. Examples:\n"
        "  terminate_instance(server_id='srv_02')\n"
        "  resize_instance(server_id='srv_05', new_size='small')\n"
        "  done()\n"
    )

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.client = None

        if OpenAI is not None and api_key:
            self.client = OpenAI(api_key=api_key, base_url=api_base_url)

    # ── Observation → prompt ──────────────────────────────────────────

    @staticmethod
    def _format_observation(obs: Observation) -> str:
        """Convert an Observation into a readable LLM prompt."""
        state = obs.cloud_state
        lines: List[str] = [
            f"=== Cloud Infrastructure (Step {state.step_count}) ===",
            f"Current daily cost: ${state.total_daily_cost:.2f}",
            f"Savings so far:     ${state.total_savings:.2f}",
            "",
            "Servers:",
        ]

        for srv in state.servers:
            status = "TERMINATED" if srv.is_terminated else "ACTIVE"
            lines.append(
                f"  {srv.server_id}  |  {status}  |  {srv.instance_type.value:>6}  |  "
                f"${srv.cost_per_day:>6.0f}/day  |  CPU {srv.cpu_usage_percent:>5.1f}%  |  "
                f"MEM {srv.memory_usage_percent:>5.1f}%  |  env={srv.environment.value}"
            )

        lines.append("")
        lines.append("Valid actions:")
        for action in obs.valid_actions[:15]:
            lines.append(f"  - {action}")
        if len(obs.valid_actions) > 15:
            lines.append(f"  ... and {len(obs.valid_actions) - 15} more")

        lines.append("")
        lines.append(
            "Choose the BEST next action to maximize cost savings while "
            "protecting production.  Respond with ONLY the action command."
        )
        return "\n".join(lines)

    # ── Decision ──────────────────────────────────────────────────────

    def decide(self, obs: Observation) -> str:
        """Ask the LLM for the next action.  Falls back to done() on error."""
        if self.client is None:
            return self._heuristic_decide(obs)

        prompt = self._format_observation(obs)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=120,
            )
            raw = response.choices[0].message.content.strip()
            return self._clean_action(raw, obs.valid_actions)
        except Exception as e:
            print(f"[WARN] LLM call failed: {e}", file=sys.stderr)
            return self._heuristic_decide(obs)

    # ── Heuristic fallback ────────────────────────────────────────────

    @staticmethod
    def _heuristic_decide(obs: Observation) -> str:
        """
        Simple rule-based fallback when no LLM is available.

        Priority order:
          1. Terminate idle (0% CPU) non-production servers
          2. Resize low-usage (<10% CPU) non-production large/xlarge → small
          3. done()
        """
        state = obs.cloud_state

        # 1. Terminate idle servers
        for srv in state.servers:
            if (not srv.is_terminated
                    and srv.environment.value != "production"
                    and srv.cpu_usage_percent == 0.0):
                return f"terminate_instance(server_id='{srv.server_id}')"

        # 2. Resize over-provisioned servers
        for srv in state.servers:
            if (not srv.is_terminated
                    and srv.environment.value != "production"
                    and srv.cpu_usage_percent < 10.0
                    and srv.instance_type.value in ("large", "xlarge")):
                return f"resize_instance(server_id='{srv.server_id}', new_size='small')"

        # 3. Nothing left to optimise
        return "done()"

    # ── Response cleaning ─────────────────────────────────────────────

    @staticmethod
    def _clean_action(raw: str, valid_actions: List[str]) -> str:
        """Extract a valid action string from raw LLM output."""
        # Try exact match first
        raw_stripped = raw.strip().strip("`").strip()
        if raw_stripped in valid_actions:
            return raw_stripped

        # Try regex extraction
        term_match = re.search(r"terminate_instance\([^)]+\)", raw)
        if term_match and term_match.group(0) in valid_actions:
            return term_match.group(0)

        resize_match = re.search(r"resize_instance\([^)]+\)", raw)
        if resize_match and resize_match.group(0) in valid_actions:
            return resize_match.group(0)

        if "done()" in raw:
            return "done()"

        # Last resort: find any valid action mentioned
        for va in valid_actions:
            if va in raw:
                return va

        return "done()"


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def run_single_task(task_id: str, agent: FinOpsAgent) -> float:
    """Run the agent on a single task and return the final score."""
    env = CloudFinOpsEnv(task_id=task_id)
    obs = env.reset()
    done = False
    step = 0
    all_rewards: List[float] = []

    print(f"[START] task={task_id} env=cloud_finops_optimizer model={agent.model_name}")

    while not done and step < env.max_steps:
        step += 1
        action = agent.decide(obs)
        obs, reward, done, info = env.step(action)

        all_rewards.append(reward.value)
        error_msg = info.get("error")
        error_str = f'"{error_msg}"' if error_msg else "null"

        print(
            f"[STEP] step={step} action={action} "
            f"reward={reward.value:.4f} done={str(done).lower()} "
            f"error={error_str}"
        )

    result = env.get_final_result()
    rewards_str = ",".join(f"{r:.4f}" for r in all_rewards)
    print(
        f"[END] task={task_id} success={str(result.success).lower()} "
        f"steps={result.steps} score={result.score:.4f} "
        f"savings=${result.total_savings:.2f} "
        f"violations={result.production_violations} "
        f"rewards={rewards_str}"
    )
    print()
    return result.score


def main():
    """Run baseline inference across all three tasks."""
    # Read configuration from environment variables
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

    print("=" * 70)
    print("  Cloud FinOps Cost Optimizer — Baseline Inference")
    print(f"  Model:    {model_name}")
    print(f"  API Base: {api_base_url}")
    print(f"  API Key:  {'***' + api_key[-4:] if len(api_key) > 4 else '(not set — using heuristic)'}")
    print("=" * 70)
    print()

    agent = FinOpsAgent(
        model_name=model_name,
        api_base_url=api_base_url,
        api_key=api_key if api_key else None,
    )

    task_ids = ["idle_killer", "rightsizer", "minefield"]
    scores: dict = {}

    for task_id in task_ids:
        score = run_single_task(task_id, agent)
        scores[task_id] = score

    # Summary
    print("=" * 70)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 70)
    for tid, sc in scores.items():
        print(f"  {tid:<20s}  score = {sc:.4f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':<20s}  score = {avg:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
