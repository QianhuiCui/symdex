# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym
from symdex.env.tasks.manager_based_env_cfg import BaseEnvCfg

# Register Gym environments.
##

from .tasks.Handover.env_cfg import HandoverEnvCfg
gym.register(
    id="HandoverEnv-v0",
    entry_point="symdex.env.tasks.Handover.env:HandoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandoverEnvCfg,
    },
)   

from .tasks.StirBowl.env_cfg import StirBowlEnvCfg
gym.register(
    id="StirBowlEnv-v0",
    entry_point="symdex.env.tasks.StirBowl.env:StirBowlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StirBowlEnvCfg,
    },
)

from .tasks.InsertDrawer.env_cfg import InsertDrawerEnvCfg
gym.register(
    id="InsertDrawerEnv-v0",
    entry_point="symdex.env.tasks.InsertDrawer.env:InsertDrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": InsertDrawerEnvCfg,
    },
)

from .tasks.BoxLift.env_cfg import BoxLiftEnvCfg
gym.register(
    id="BoxLiftEnv-v0",
    entry_point="symdex.env.tasks.BoxLift.env:BoxLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BoxLiftEnvCfg,
    },
)

from .tasks.PickObject.env_cfg import PickObjectEnvCfg
gym.register(
    id="PickObjectEnv-v0",
    entry_point="symdex.env.tasks.PickObject.env:PickObjectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PickObjectEnvCfg,
    },
)

from .tasks.Threading.env_cfg import ThreadingEnvCfg
gym.register(
    id="ThreadingEnv-v0",
    entry_point="symdex.env.tasks.Threading.env:ThreadingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ThreadingEnvCfg,
    },
)

from .tasks.Pouring.env_cfg import PouringEnvCfg
gym.register(
    id="PouringEnv-v0",
    entry_point="symdex.env.tasks.Pouring.env:PouringEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PouringEnvCfg,
    },
)