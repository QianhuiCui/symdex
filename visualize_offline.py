from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import hydra
import torch
import gymnasium as gym
from omegaconf import DictConfig

import symdex
from symdex.algo import alg_name_to_path
from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg, load_class_from_path
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper
from symdex.utils.offline_buffer import OfflineBuffer


@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    cfg, env_cfg = preprocess_cfg(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device, clip_obs=50.0)

    replay_buffer = OfflineBuffer(device=cfg.device, use_vision=cfg.algo.observation.vision, use_pc=cfg.algo.observation.pc)
    replay_buffer.load_from_dir(cfg.algo.offline.data_dir, pattern=cfg.algo.offline.pattern)

    max_action = env.unwrapped.action_space.high[0]
    print(f"[Env] action_space high: {max_action:.6f}")

    if cfg.algo.normalize_states:
        state_mean, state_std = replay_buffer.normalize_states()
    else:
        state_mean, state_std = None, None
    algo_name = cfg.algo.name
    algo_class = load_class_from_path(algo_name, alg_name_to_path[algo_name])
    policy = algo_class(
        state_dim = replay_buffer.state_dim,
        action_dim = replay_buffer.action_dim,
        max_action = max_action,
		# max_action = cfg.algo.offline.max_action,
        # max_action = estimated_max_action,
        device = cfg.device,
        use_vision = cfg.algo.observation.vision,
        use_pc = cfg.algo.observation.pc,
		discount=cfg.algo.discount,
		tau=cfg.algo.tau,
		policy_noise=cfg.algo.policy_noise,
		noise_clip=cfg.algo.noise_clip,
		policy_freq=cfg.algo.policy_freq,
		alpha=cfg.algo.alpha,
        actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr,
    )
    assert cfg.algo.checkpoint.load_path is not None, "Please set algo.checkpoint.load_path"
    policy.load(cfg.algo.checkpoint.load_path)

    obs, _ = env.reset()