from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from functools import reduce

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int) -> torch.Tensor:
    object_ids = [0, 1]
    command_names = ["target_pos_cup", "target_pos_bowl"]
    for i in range(len(object_ids)):
        command_term = env.command_manager.get_term(command_names[i])
        if_success = (command_term.metrics["consecutive_success"] >= 1)
        idx = torch.where(if_success)[0]
        fail_idx = torch.where(~if_success)[0]
        env.object_success_tracker[object_ids[i]][idx] += 1
        env.object_success_tracker[object_ids[i]][fail_idx] = 0
    
    # # check if ball is in the bowl
    # bowl = env.scene["object_0"]
    # ball = env.scene["object_2"]
    # ball_in_bowl = torch.norm(ball.data.root_pos_w[:, :3] - bowl.data.root_pos_w[:, :3], dim=-1) < 0.15
    # check if all objects are in the tote and track the success steps
    success = reduce(torch.logical_and, [env.object_success_tracker[0] >= 1, env.object_success_tracker[1] >= 1]).bool()
    success_idx = torch.where(success)[0]
    unsuccess_idx = torch.where(~success)[0]
    env.success_tracker_step[success_idx] += 1
    env.success_tracker_step[unsuccess_idx] = 0
    
    success = env.success_tracker_step >= num_success
    env.success_tracker = success.float()
    return success

def generated_commands_right(env: ManagerBasedRLEnv):
    waiting_term = env.command_manager.get_term("waiting_pos_cup")
    if_success = torch.where(waiting_term.metrics["consecutive_success"] >= 1)[0]
    env.cup_reach_first_target[if_success] += 1.0

    waiting_pos = env.command_manager.get_command("waiting_pos_cup")[:, :7].clone()
    target_pos = env.command_manager.get_command("target_pos_cup")[:, :7].clone()
    if not hasattr(env, "object_success_tracker"):
        return torch.zeros((env.num_envs, 3), device=env.device)
    both_success = (env.object_success_tracker[1] >= 1) * (env.cup_reach_first_target >= 5)
    both_success_idx = torch.where(both_success)[0]
    waiting_pos[both_success_idx] = target_pos[both_success_idx]
    return waiting_pos

def object_success(env: ManagerBasedRLEnv, object_id: int, one_shot: bool = False):
    if one_shot:
        rew = (env.object_success_tracker[object_id] == 3).float() * (env.object_bonus_tracker[object_id] == 0).float()
        bonus_idx = torch.where(rew > 0.0)[0]
        env.object_bonus_tracker[object_id][bonus_idx] += 1.0
        return rew
    else:
        return (env.object_success_tracker[object_id] >= 1).float()


def cmd_wait(env: ManagerBasedRLEnv):
    rew = (env.cup_reach_first_target >= 5).float() * (env.object_bonus_tracker[0] == 0).float()
    bonus_idx = torch.where(rew > 0.0)[0]
    env.object_bonus_tracker[0][bonus_idx] += 1.0
    return rew

def generated_commands_right_weights(env: ManagerBasedRLEnv):
    weights = torch.ones(env.num_envs, device=env.device)
    waiting_term = env.command_manager.get_term("waiting_pos_cup")
    if_success = torch.where(waiting_term.metrics["consecutive_success"] >= 1)[0]
    env.cup_reach_first_target[if_success] += 1.0

    waiting_pos = env.command_manager.get_command("waiting_pos_cup")[:, :7].clone()
    target_pos = env.command_manager.get_command("target_pos_cup")[:, :7].clone()
    if not hasattr(env, "object_success_tracker"):
        waiting_pos = torch.zeros((env.num_envs, 3), device=env.device)
    else:
        both_success = (env.object_success_tracker[1] >= 1) * (env.cup_reach_first_target >= 5)
        both_success_idx = torch.where(both_success)[0]
        waiting_pos[both_success_idx] = target_pos[both_success_idx]
        weights[both_success_idx] = 2.0

    return [waiting_pos, weights]