
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import hydra
from omegaconf import DictConfig
from loguru import logger
import gymnasium as gym

import symdex
from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, load_class_from_path, preprocess_cfg, Tracker
from symdex.algo.network import model_name_to_path
from symdex.utils.model_util import load_model
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper
from symdex.utils.symmetry import SymmetryManager

@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    cfg, env_cfg = preprocess_cfg(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)
    device = torch.device(cfg.device)
    act_class = load_class_from_path(cfg.algo.act_class,
                                            model_name_to_path[cfg.algo.act_class])
    
    multi_agent_cfg = cfg.task.multi.SYMDEX
    symmetry_cfg = cfg.task.symmetry.SYMDEX
    action_dim = [22, 22]
    actor = []
    for k in range(len(multi_agent_cfg.single_agent_obs_dim)):
        if "Equivariant" in cfg.algo.act_class:
            cur_actor = act_class(env.unwrapped.G, symmetry_cfg.actor_input_fields[k], symmetry_cfg.actor_output_fields[k], multi_agent_cfg.single_agent_obs_dim[k], action_dim[k]).to(device)
        else:
            cur_actor = act_class(cfg.task.multi.SYMDEX.single_agent_obs_dim[k], cfg.task.multi.SYMDEX.single_agent_action_dim).to(device)
        load_model(cur_actor, f"actor_{k}", cfg.artifact)
        actor.append(cur_actor)    
    symmetry_manager = SymmetryManager(cfg=multi_agent_cfg, symmetric_envs=cfg.task.symmetry.symmetric_envs)

    return_tracker = Tracker(cfg.num_envs)
    step_tracker = Tracker(cfg.num_envs)
    current_rewards = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
    current_lengths = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
    
    if cfg.task.randomize.eval:
        env.unwrapped.update_randomization(1.0)
    else:
        env.unwrapped.update_randomization(0.0)
    obs, _ = env.reset()
    for _ in range(env.max_episode_length * 100):  # run 100 episodes
        with torch.no_grad():
            obs_list = symmetry_manager.get_multi_agent_obs(obs, env.unwrapped.symmetry_tracker)
            action1 = actor[0](obs_list[0], sample=False)
            action2 = actor[1](obs_list[1], sample=False)
            action = symmetry_manager.get_execute_action(action1, action2, env.unwrapped.symmetry_tracker)
        next_obs, reward, done, info = env.step(action)
        current_rewards += reward
        current_lengths += 1
        env_done_indices = torch.where(done)[0]
        return_tracker.update(current_rewards[env_done_indices])
        step_tracker.update(current_lengths[env_done_indices])
        current_rewards[env_done_indices] = 0
        current_lengths[env_done_indices] = 0
        obs = next_obs

    r_exp = return_tracker.mean()
    step_exp = step_tracker.mean()
    logger.warning(f"Cumulative return: {r_exp}, Episode length: {step_exp}")


if __name__ == '__main__':
    main()