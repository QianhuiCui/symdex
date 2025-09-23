from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from .reach_command import TargetPositionCommand

@configclass
class TargetPositionCommandCfg(CommandTermCfg):
    class_type: type = TargetPositionCommand

    asset_cfg = None

    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time

    pose_range: dict = None
    """The goal pose range."""

    target_link: str = None
    """The target link to reach."""

    success_threshold: float = MISSING
    """Threshold for the position error to consider the goal position to be reached."""

    success_threshold_orient: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""

    update_goal_on_success: bool = MISSING
    """Whether to update the goal when the goal is reached."""

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path=f"/Visuals/Command/goal_marker_{target_link}",
        markers={
            "goal": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
        },
    )
    """Configuration for the visualization markers."""

    return_type: str = None
