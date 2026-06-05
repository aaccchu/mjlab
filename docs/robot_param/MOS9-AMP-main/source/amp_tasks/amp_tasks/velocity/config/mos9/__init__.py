import gymnasium as gym


def _register_mos9_task(task_id: str, env_cfg: str, runner_cfg: str):
  gym.register(
    id=task_id,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
      "env_cfg_entry_point": f"{__name__}.velocity_env_param_cfg:{env_cfg}",
      "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
    },
  )


_register_mos9_task(
  "AMP_MOS9_Velocity",
  "MOS9VelocityAMPEnvCfg",
  "MOS9VelocityAMPRunnerCfg",
)

_register_mos9_task(
  "AMP_MOS9_Velocity_LessMotion",
  "MOS9VelocityAMPEnvCfg",
  "MOS9VelocityAMPLessMotionRunnerCfg",
)

_register_mos9_task(
  "AMP_MOS9_Velocity_LessMotion_NoCurriculum",
  "MOS9VelocityAMPEnvCfgNoCurriculum",
  "MOS9VelocityAMPLessMotionRunnerCfg",
)

_register_mos9_task(
  "AMP_MOS9_Velocity_ModifiedMotion_NoCurriculum",
  "MOS9VelocityAMPEnvCfgNoCurriculum",
  "MOS9VelocityAMPModifiedMotionRunnerCfg",
)


_register_mos9_task(
  "AMP_MOS9_V6",
  "MOS9EnvCfgV6",
  "MOS9AMPRunnerCfgV6",
)

_register_mos9_task(
  "AMP_MOS9_V6_ALIAS",
  "MOS9EnvCfgV6",
  "MOS9AMPRunnerCfgV6Alias",
)

_register_mos9_task(
  "AMP_MOS9_V7",
  "MOS9EnvCfgV7",
  "MOS9AMPRunnerCfgV6",
)

_register_mos9_task(
  "AMP_MOS9_V7_ALIAS",
  "MOS9EnvCfgV7",
  "MOS9AMPRunnerCfgV6Alias",
)


_register_mos9_task(
  "AMP_MOS9_V11",
  "MOS9EnvCfgV11",
  "MOS9AMPRunnerCfgV11",
)


_register_mos9_task(
  "AMP_MOS9_V8",
  "MOS9EnvCfgV8",
  "MOS9AMPRunnerCfgV6Alias",
)

_register_mos9_task(
  "AMP_MOS9_V9",
  "MOS9EnvCfgV9",
  "MOS9AMPRunnerCfgV6Alias",
)
_register_mos9_task(
  "AMP_MOS9_V9A",
  "MOS9EnvCfgV9",
  "MOS9AMPRunnerCfgV9Alias",
)

_register_mos9_task(
  "AMP_MOS9_V9B",
  "MOS9EnvCfgV9",
  "MOS9AMPRunnerCfgV9B",
)

_register_mos9_task(
  "AMP_MOS9_V10",
  "MOS9EnvCfgV10",
  "MOS9AMPRunnerCfgV9Alias",
)
