from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import hydra
import wandb
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace

import symdex
from symdex.algo import alg_name_to_path
from symdex.utils.common import init_wandb, set_random_seed, capture_keyboard_interrupt, preprocess_cfg, load_class_from_path
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.evaluator_td3bc import EvaluatorTD3BC
from symdex.utils.rl_env_wrapper import VecEnvWrapper
from symdex.utils.offline_buffer import OfflineBuffer


@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    cfg, env_cfg = preprocess_cfg(cfg)
    wandb_run = init_wandb(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device, clip_obs=50.0)
    # DEBUG
    # print("cfg.task.cam.resolution:", cfg.task.cam.resolution)
    # print("env_cfg camera width:", env_cfg.scene.cam_1.width)
    # print("env_cfg camera height:", env_cfg.scene.cam_1.height)

    replay_buffer = OfflineBuffer(device=cfg.device, use_vision=cfg.algo.observation.vision, use_pc=cfg.algo.observation.pc)
    replay_buffer.load_from_dir(cfg.algo.offline.data_dir, pattern=cfg.algo.offline.pattern)

    # DEBUG 
    # action_stats = replay_buffer.action_stats()
    # print(f"[OfflineBuffer] action stats: {action_stats}")

    # Auto max_action
    # estimated_max_action = replay_buffer.estimate_max_action(quantile=0.99)
    # print(f"[OfflineBuffer] estimated max_action (q=0.99): {estimated_max_action:.6f}")
    # max_action = env.unwrapped.action_space.high
    # print(f"[Env] action_space high: {max_action:.6f}")

    if cfg.algo.normalize_states:
        state_mean, state_std = replay_buffer.normalize_states()
    else:
        state_mean, state_std = None, None
    algo_name = cfg.algo.name
    algo_class = load_class_from_path(algo_name, alg_name_to_path[algo_name])

    multi_td3bc = cfg.task.multi.TD3BC
    actor_cfg = SimpleNamespace(
        single_agent_obs_dim=list(multi_td3bc.actor_obs_dim),
        single_agent_obs_idx=multi_td3bc.actor_obs_idx,
    )
    critic_cfg = SimpleNamespace(
        single_agent_obs_dim=list(multi_td3bc.single_agent_obs_dim),
        single_agent_obs_idx=multi_td3bc.single_agent_obs_idx,
    )

    policy = algo_class(
        actor_cfg=actor_cfg,
        critic_cfg=critic_cfg,
        action_dim=replay_buffer.action_dim,
        max_action=cfg.algo.offline.max_action,
        device=cfg.device,
        discount=cfg.algo.discount,
        tau=cfg.algo.tau,
        policy_noise=cfg.algo.policy_noise,
        noise_clip=cfg.algo.noise_clip,
        policy_freq=cfg.algo.policy_freq,
        alpha=cfg.algo.alpha,
        actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr,
        reward_scale=cfg.algo.reward_scale,
        pc_max_points=cfg.algo.get("pc_max_points", 512),
        pretrain_ckpt_path=cfg.algo.checkpoint.get("pretrain_encoder_path", None),
    )
    
    if cfg.algo.checkpoint.load_path is not None:
        policy.load(cfg.algo.checkpoint.load_path)

    global_steps = 0
    success_max = float('-inf')
    evaluator = EvaluatorTD3BC(cfg=cfg, env_cfg=env_cfg, env=env, wandb_run=wandb_run, state_mean=state_mean, state_std=state_std)

    randomization_state, best_so_far = None, None
    if cfg.task.randomize.eval:
        env.unwrapped.update_randomization(1.0)
    else:
        env.unwrapped.update_randomization(0.0)
    
    for step in range(1, cfg.algo.max_train_steps + 1):
        global_steps = step
        log_diagnostics = step % cfg.algo.log_freq == 0
        log_info = policy.train(replay_buffer, cfg.algo.batch_size, log_diagnostics=log_diagnostics)

        if step % cfg.algo.log_freq == 0:
            log_info["global_steps"] = global_steps
            log_info["dataset/size"] = replay_buffer.size
            # log_info["dataset/action_abs_p95"] = action_stats["abs_p95"]
            # log_info["dataset/action_abs_p99"] = action_stats["abs_p99"]
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
        
        if step % cfg.algo.eval_freq == 0:
            return_dict, success_max = evaluator.eval_policy(policy=policy, success_max=success_max)
            wandb.log(return_dict, step=global_steps)
            if cfg.task.randomize.enable:
                # domain randomization
                randomization_state, curriculum_state, best_so_far = env.unwrapped.update_randomization(return_dict["eval/success_rate"])
                success_max = float('-inf')
        
        if evaluator.check_if_should_stop(global_steps):
            break

    policy.save(f"{wandb_run.dir}/model_latest.pth")
    print(f"[INFO] Saved latest model to {wandb_run.dir}/model_latest.pth")
    # policy.save(f"{wandb_run.dir}/model_final.pth")
    if state_mean is not None and state_std is not None:
        torch.save({
            "state_mean": state_mean,
            "state_std": state_std,
            },f"{wandb_run.dir}/state_norm.pth",)
    OmegaConf.save(cfg, f"{wandb_run.dir}/resolved_cfg.yaml")


if __name__ == '__main__':
    main()