from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

from itertools import count
import hydra
import wandb
import gymnasium as gym
from omegaconf import DictConfig

import symdex
from symdex.algo import alg_name_to_path
from symdex.utils.common import init_wandb, load_class_from_path, set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.utils.model_util import load_model
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.evaluator import Evaluator
from symdex.utils.rl_env_wrapper import VecEnvWrapper


@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    cfg, env_cfg = preprocess_cfg(cfg)
    wandb_run = init_wandb(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device, clip_obs=50.0)

    algo_name = cfg.algo.name
    if 'Agent' not in algo_name:
        algo_name = 'Agent' + algo_name
    agent_class = load_class_from_path(algo_name, alg_name_to_path[algo_name])
    agent = agent_class(env=env, 
                        cfg=cfg, 
                        obs_dim=cfg.task.multi.single_agent_obs_dim if cfg.algo.multi_agent else env.observation_space.shape[-1], 
                        action_dim=cfg.task.multi.single_agent_action_dim if cfg.algo.multi_agent else env.action_space.shape[-1])

    if cfg.artifact is not None:
        load_model(agent.actor, "actor_0", cfg.artifact)
        load_model(agent.actor_left, "actor_1", cfg.artifact)
        if cfg.algo.obs_norm:
            load_model(agent.obs_rms, "obs_rms", cfg.artifact)

    global_steps = 0
    success_max = float('-inf')
    evaluator = Evaluator(cfg=cfg, env_cfg=env_cfg, env=env, wandb_run=wandb_run)

    randomization_state, best_so_far = None, None
    if cfg.task.randomize.eval:
        env.unwrapped.update_randomization(1.0)
    else:
        env.unwrapped.update_randomization(0.0)
    agent.reset_agent()

    for iter_t in count():
        if iter_t % cfg.algo.eval_freq == 0:
            return_dict, success_max = evaluator.eval_policy([agent.actor, agent.actor_left] if cfg.algo.multi_agent else agent.actor, 
                                                        [agent.critic, agent.critic_left] if cfg.algo.multi_agent else agent.critic,
                                                        algo_multi_cfg=agent.algo_multi_cfg if cfg.algo.multi_agent else None,
                                                        normalizer=agent.obs_rms,
                                                        success_max=success_max
                                                        )
            wandb.log(return_dict, step=global_steps)
            agent.reset_agent()
            
        trajectory, steps = agent.explore_env(env, cfg.algo.horizon_len, random=False)
        global_steps += steps
        log_info = agent.update_net(trajectory)

        if iter_t % cfg.algo.log_freq == 0:
            log_info['global_steps'] = global_steps
            for key in agent.detailed_tracker.keys():
                log_info[f'Rewards/{key}'] = agent.detailed_tracker[key].mean()
                
            if randomization_state is not None:
                for param in randomization_state.keys():
                    log_info[f'Randomization/{param}_sigma'] = randomization_state[param]['sigma']
                for parm in curriculum_state.keys():
                    val = curriculum_state[parm]
                    if isinstance(val, DictConfig):
                        continue
                    log_info[f'Curriculum/{parm}_value'] = val[min(best_so_far, len(val) - 1)]
                log_info['Randomization/best_so_far'] = best_so_far

            wandb.log(log_info, step=global_steps)

        if iter_t % cfg.task.randomize.update_freq == 0 and cfg.task.randomize.enable:
            # domain randomization
            randomization_state, curriculum_state, best_so_far = env.unwrapped.update_randomization(log_info['train/success_rate'])
            success_max = float('-inf')
                
        if evaluator.check_if_should_stop(global_steps):
            break


if __name__ == '__main__':
    main()
