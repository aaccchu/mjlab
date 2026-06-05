from isaaclab.utils import configclass

from ...mdp.amp_obs_grp import AMPObsBaiscCfg
from .velocity_env_cfg import RobotEnvCfg


@configclass
class G1VelocityAMPEnvCfg(RobotEnvCfg):
  def __post_init__(self):
    super().__post_init__()
    self.observations.amp = AMPObsBaiscCfg()
