"""pose2arm RL task — registers Isaac-OpenArmRetarget-Direct-v0 with gym."""
import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-OpenArmRetarget-Direct-v0",
    entry_point=f"{__name__}.env:OpenArmRetargetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:OpenArmRetargetEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
