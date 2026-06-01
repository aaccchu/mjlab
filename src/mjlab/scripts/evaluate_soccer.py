"""Evaluate a trained soccer dribbling policy and compute metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro

import mjlab
import mjlab.tasks  # noqa: F401 — populate registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.dribble_command import DribbleCommand
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Soccer-Unitree-G1"


@dataclass(frozen=True)
class EvalConfig:
  """Configuration for soccer policy evaluation."""

  checkpoint_file: str
  num_envs: int = 512
  seed: int = 42
  max_episodes: int = 512
  device: str | None = None
  output_csv: str | None = None


def _ci95(arr: np.ndarray) -> tuple[float, float, float]:
  """Return (mean, lower, upper) for 95% CI via bootstrap."""
  mean = float(np.mean(arr))
  if len(arr) < 2:
    return mean, mean, mean
  se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
  return mean, mean - 1.96 * se, mean + 1.96 * se


def run_evaluate(cfg: EvalConfig) -> dict[str, float]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(TASK_ID, play=False)
  agent_cfg = load_rl_cfg(TASK_ID)

  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  env_cfg.auto_reset = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env.seed(cfg.seed)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    cfg.checkpoint_file,
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  command = cast(
    DribbleCommand,
    env.unwrapped.command_manager.get_term("dribble"),
  )

  successes: list[float] = []
  time_to_goals: list[float] = []
  possessions: list[float] = []
  ball_errors: list[float] = []
  falls: list[float] = []
  completed = 0

  obs = env.get_observations()
  raw_env = env.unwrapped

  while completed < cfg.max_episodes:
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)

    done_mask = dones.bool()
    if not done_mask.any():
      continue

    done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
    for idx in done_ids:
      i = int(idx.item())
      completed += 1
      if completed > cfg.max_episodes:
        break

      sc = command.metrics["step_count"][i].item()
      ttg = command.metrics["time_to_goal"][i].item()
      poss = command.metrics["possession"][i].item()
      bte = command.metrics["ball_to_target_error"][i].item()
      es = command.metrics["episode_success"][i].item()

      successes.append(float(es > 0))
      ball_errors.append(bte)
      if es > 0 and ttg > 0:
        time_to_goals.append(ttg)
      if sc > 0:
        possessions.append(poss / sc)

      tm = raw_env.termination_manager
      fell = bool(tm.terminated[i]) and not bool(tm.time_outs[i])
      falls.append(float(fell))

    raw_env.reset(env_ids=done_ids)
    reset_obs = raw_env.observation_manager.compute()
    obs["actor"][done_ids] = reset_obs["actor"][done_ids]  # type: ignore[index]

  env.close()

  s_arr = np.array(successes)
  f_arr = np.array(falls)
  p_arr = np.array(possessions) if possessions else np.array([0.0])
  b_arr = np.array(ball_errors)
  t_arr = np.array(time_to_goals) if time_to_goals else np.array([0.0])

  ci_results: dict[str, tuple[float, float, float]] = {
    "success_rate": _ci95(s_arr),
    "fall_rate": _ci95(f_arr),
    "possession_rate": _ci95(p_arr),
    "ball_to_target_error": _ci95(b_arr),
    "time_to_goal": _ci95(t_arr),
  }

  print(f"\n{'Metric':<25} {'Mean':>8} {'CI_low':>8} {'CI_high':>8}")
  print("-" * 55)
  for k, (m, lo, hi) in ci_results.items():
    print(f"{k:<25} {m:>8.4f} {lo:>8.4f} {hi:>8.4f}")
  print(f"{'n_episodes':<25} {completed:>8}")

  if cfg.output_csv:
    path = Path(cfg.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
      f.write("metric,mean,ci_low,ci_high\n")
      for k, (m, lo, hi) in ci_results.items():
        f.write(f"{k},{m:.6f},{lo:.6f},{hi:.6f}\n")
      f.write(f"n_episodes,{completed},,\n")
    print(f"\nResults saved to {path}")

  return {k: v[0] for k, v in ci_results.items()}


def main():
  maybe_print_top_level_help("evaluate-soccer")
  args = tyro.cli(EvalConfig, config=mjlab.TYRO_FLAGS)
  run_evaluate(args)


if __name__ == "__main__":
  main()
