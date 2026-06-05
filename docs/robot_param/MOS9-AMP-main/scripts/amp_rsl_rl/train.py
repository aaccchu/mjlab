import argparse
import os

from isaaclab import __version__ as omni_isaac_lab_version

assert omni_isaac_lab_version > "0.21.0"
from isaaclab.app import AppLauncher

# local imports
import argtool as rsl_arg_cli  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--alg", type=str, default="PPO", help="Name of the algorithm.")
parser.add_argument(
  "--cfg", type=str, default=None, help="Directly using the target cfg object."
)
parser.add_argument(
  "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
  "--seed", type=int, default=42, help="Seed used for the environment"
)
parser.add_argument(
  "--replicate",
  type=str,
  default=None,
  help="Replicate old experiment with same configuration.",
)

parser.add_argument("--rldevice", type=str, default="cuda:0", help="Device for rl")
parser.add_argument(
  "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
  "--video_length",
  type=int,
  default=200,
  help="Length of the recorded video (in steps).",
)
parser.add_argument(
  "--video_interval",
  type=int,
  default=2000,
  help="Interval between video recordings (in steps).",
)
parser.add_argument(
  "--local", action="store_true", default=False, help="Using asset in local buffer"
)

# append RSL-RL cli arguments
rsl_arg_cli.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()


reward_overrides = {}
event_overrides = {}
agent_overrides = {}


def _parse_cli_value(val_str: str):
  if "," in val_str:
    items = [v.strip() for v in val_str.split(",")]
    try:
      return tuple(float(x) for x in items)
    except ValueError:
      return tuple(items)

  lower = val_str.lower()
  if lower == "true":
    return True
  if lower == "false":
    return False

  try:
    return int(val_str)
  except ValueError:
    try:
      return float(val_str)
    except ValueError:
      return val_str


def _set_by_path(root, path: str, value):
  parts = path.split(".")
  current = root
  for part in parts[:-1]:
    if isinstance(current, dict):
      if part not in current:
        raise KeyError(f"Key '{part}' not found while setting '{path}'")
      current = current[part]
    else:
      if not hasattr(current, part):
        raise AttributeError(f"Attribute '{part}' not found while setting '{path}'")
      current = getattr(current, part)

  leaf = parts[-1]
  if isinstance(current, dict):
    if leaf not in current:
      raise KeyError(f"Key '{leaf}' not found while setting '{path}'")
    current[leaf] = value
  else:
    if not hasattr(current, leaf):
      raise AttributeError(f"Attribute '{leaf}' not found while setting '{path}'")
    setattr(current, leaf, value)


def _get_by_path(root, path: str):
  parts = path.split(".")
  current = root
  for part in parts:
    if isinstance(current, dict):
      current = current[part]
    else:
      current = getattr(current, part)
  return current


for arg in unknown_args:
  if "=" not in arg:
    print(f"[WARNING] Ignore unknown arg without '=': {arg}")
    continue

  key, value = arg.split("=", 1)
  parsed_value = _parse_cli_value(value)

  if key.startswith("--reward_"):
    reward_name = key.replace("--reward_", "", 1)
    reward_overrides[reward_name] = parsed_value
  elif key.startswith("--event_"):
    event_path = key.replace("--event_", "", 1)
    event_overrides[event_path] = parsed_value
  elif key.startswith("--agent_"):
    agent_path = key.replace("--agent_", "", 1)
    agent_overrides[agent_path] = parsed_value
  else:
    print(f"[WARNING] Unknown override key: {key}")

if reward_overrides:
  print(f"[INFO] CLI reward overrides: {reward_overrides}")
if event_overrides:
  print(f"[INFO] CLI event overrides: {event_overrides}")
if agent_overrides:
  print(f"[INFO] CLI agent overrides: {agent_overrides}")


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from amp_tasks.amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _apply_reward_overrides(env_cfg, overrides: dict):
  if not overrides:
    return
  if not hasattr(env_cfg, "rewards"):
    print("[WARNING] env_cfg has no rewards section. Ignore reward overrides.")
    return

  print("[INFO] Applying reward overrides...")
  for reward_path, reward_val in overrides.items():
    # Handle parameter overrides (e.g. track_lin_vel_xy.params.std)
    if ".params." in reward_path:
      reward_name, param_path = reward_path.split(".params.", 1)
      if not hasattr(env_cfg.rewards, reward_name):
        print(f"[WARNING] Reward term '{reward_name}' not found.")
        continue

      reward_term = getattr(env_cfg.rewards, reward_name)
      if not hasattr(reward_term, "params"):
        print(f"[WARNING] Reward term '{reward_name}' has no params field.")
        continue

      try:
        old_val = _get_by_path(reward_term.params, param_path)
        _set_by_path(reward_term.params, param_path, reward_val)
        print(
          f"    -> rewards.{reward_name}.params.{param_path}: {old_val} -> {reward_val}"
        )
      except (KeyError, AttributeError) as e:
        print(f"[WARNING] Failed to set param for {reward_name}: {e}")
      continue

    # Backward compatibility for weight overrides (e.g. track_lin_vel_xy)
    reward_name = reward_path
    if not hasattr(env_cfg.rewards, reward_name):
      print(f"[WARNING] Reward term '{reward_name}' not found.")
      continue
    reward_term = getattr(env_cfg.rewards, reward_name)
    if not hasattr(reward_term, "weight"):
      print(f"[WARNING] Reward term '{reward_name}' has no weight field.")
      continue
    old_weight = reward_term.weight
    reward_term.weight = reward_val
    print(f"    -> rewards.{reward_name}.weight: {old_weight} -> {reward_val}")


def _apply_event_overrides(env_cfg, overrides: dict):
  if not overrides:
    return
  if not hasattr(env_cfg, "events"):
    print("[WARNING] env_cfg has no events section. Ignore event overrides.")
    return

  print("[INFO] Applying event overrides...")
  for event_path, val in overrides.items():
    try:
      old_val = _get_by_path(env_cfg.events, event_path)
      _set_by_path(env_cfg.events, event_path, val)
      print(f"    -> events.{event_path}: {old_val} -> {val}")
    except (KeyError, AttributeError) as e:
      print(f"[WARNING] {e}")


def _apply_agent_overrides(agent_cfg, overrides: dict):
  if not overrides:
    return

  print("[INFO] Applying agent overrides...")
  for agent_path, val in overrides.items():
    try:
      old_val = _get_by_path(agent_cfg, agent_path)
      _set_by_path(agent_cfg, agent_path, val)
      print(f"    -> agent.{agent_path}: {old_val} -> {val}")
    except (KeyError, AttributeError) as e:
      print(f"[WARNING] {e}")


def main():
  task_name, env_cfg, agent_cfg, log_dir = rsl_arg_cli.make_cfgs(
    args_cli, parse_env_cfg, None
  )

  _apply_reward_overrides(env_cfg, reward_overrides)
  _apply_event_overrides(env_cfg, event_overrides)
  _apply_agent_overrides(agent_cfg, agent_overrides)

  env_cfg.sim.device = args_cli.device
  env_cfg.seed = args_cli.seed

  # create isaac environment
  env = gym.make(
    task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
  )

  # wrap for video recording
  if args_cli.video:
    video_kwargs = {
      "video_folder": os.path.join(log_dir, "videos"),
      "step_trigger": lambda step: step % args_cli.video_interval == 0,
      "video_length": args_cli.length,
      "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

  agent_cfg.seed = args_cli.seed
  agent_cfg.device = args_cli.rldevice

  env, func_runner, learn_cfg = rsl_arg_cli.prepare_wrapper(env, args_cli, agent_cfg)
  runner: AMPOnPolicyRunner = func_runner(
    env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
  )

  # save resume path before creating a new log_dir
  if getattr(agent_cfg, "resume", False):
    load_run = getattr(agent_cfg, "load_run", None)
    load_checkpoint = getattr(agent_cfg, "load_checkpoint", None)

    if load_run is None or load_checkpoint is None:
      raise ValueError(
        f"resume=True, but load_run ({load_run}) or load_checkpoint ({load_checkpoint}) is missing."
      )

    # 按你的日志目录结构拼路径
    resume_path = os.path.join(
      "logs",
      "rsl_rl",
      agent_cfg.experiment_name,
      load_run,
      f"model_{load_checkpoint}.pt",
    )
    resume_path = os.path.abspath(resume_path)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

  init_weight = getattr(agent_cfg, "init_weight", None)
  if init_weight:
    runner.load(init_weight)

  # dump the configuration into log-directory
  dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
  dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
  dump_yaml(os.path.join(log_dir, "params", "args.yaml"), vars(args_cli))
  rsl_arg_cli.dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
  rsl_arg_cli.dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
  rsl_arg_cli.dump_pickle(os.path.join(log_dir, "params", "args.pkl"), args_cli)

  # run training
  runner.learn(**learn_cfg)

  # close the simulator
  env.close()


if __name__ == "__main__":
  # run the main execution
  main()
  # close sim app
  simulation_app.close()
