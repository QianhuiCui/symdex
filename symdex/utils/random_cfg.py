from __future__ import annotations

import os
import re
import random
import json
from dataclasses import MISSING, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Optional
from collections.abc import Callable
from typing import TYPE_CHECKING

from pxr import Gf, Usd, UsdGeom, UsdShade
import omni.usd
import omni.kit.commands
import isaacsim.core.utils.prims as prim_utils
from omni.physx.scripts import utils as physx_utils
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.spawners.wrappers import spawn_multi_asset
from isaaclab.sim.spawners.wrappers import wrappers_cfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_prim
from isaaclab.sim.spawners.materials import visual_materials_cfg, visual_materials

import symdex


OBJECT: Dict[str, List[str]] = {
    "cup": [
            f"{symdex.LIB_PATH}/assets/objects/cups/cup_1/base.usd",
            f"{symdex.LIB_PATH}/assets/objects/cups/cup_2/base.usd",
            f"{symdex.LIB_PATH}/assets/objects/cups/cup_3/base.usd",
            f"{symdex.LIB_PATH}/assets/objects/cups/cup_4/base.usd",
            f"{symdex.LIB_PATH}/assets/objects/cups/cup_5/base.usd"
        ],
    "bowl": [
        f"{symdex.LIB_PATH}/assets/objects/bowls/bowl_1/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/bowls/bowl_2/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/bowls/bowl_3/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/bowls/bowl_4/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/bowls/bowl_5/base.usd"
    ],
    "bottle": [
        f"{symdex.LIB_PATH}/assets/objects/bottels/bottle_1/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/bottels/bottle_2/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/bottels/bottle_3/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/bottels/bottle_4/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/bottels/bottle_5/base.usd"
    ],
    "scanner": [
        f"{symdex.LIB_PATH}/assets/objects/scanners/scanner_1/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/scanners/scanner_2/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/scanners/scanner_3/base.usd",
        f"{symdex.LIB_PATH}/assets/objects/scanners/scanner_4/base.usd"
    ],
    "grasp": [
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_1/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_2/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_3/visual_model.usd",
        # f"{symdex.LIB_PATH}/assets/objects/grasping/object_4/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/grasp/dog_coacd.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_5/visual_model.usd",
        # f"{symdex.LIB_PATH}/assets/objects/grasping/object_6/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_7/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_8/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_9/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_10/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_11/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_12/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_13/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_14/visual_model.usd",
        # f"{symdex.LIB_PATH}/assets/objects/grasping/object_15/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_16/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_17/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_18/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_19/visual_model.usd",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_20/visual_model.usd"
    ]
}

OBJECT_CATEGORY: dict[str, str] = {
    "grasp": {
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_1/visual_model.usd": "fist",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_2/visual_model.usd": "puppy",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_3/visual_model.usd": "ox",
        # f"{symdex.LIB_PATH}/assets/objects/grasping/object_4/visual_model.usd": "dolphin",
        f"{symdex.LIB_PATH}/assets/grasp/dog_coacd.usd": "dog",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_5/visual_model.usd": "helmet",
        # f"{symdex.LIB_PATH}/assets/objects/grasping/object_6/visual_model.usd": "shark",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_7/visual_model.usd": "cat",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_8/visual_model.usd": "rhino",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_9/visual_model.usd": "kitten",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_10/visual_model.usd": "car",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_11/visual_model.usd": "robot",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_12/visual_model.usd": "piggy",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_13/visual_model.usd": "squirrel",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_14/visual_model.usd": "alien",
        # f"{symdex.LIB_PATH}/assets/objects/grasping/object_15/visual_model.usd": "whale",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_16/visual_model.usd": "headset",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_17/visual_model.usd": "truck",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_18/visual_model.usd": "calf",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_19/visual_model.usd": "cow",
        f"{symdex.LIB_PATH}/assets/objects/grasping/object_20/visual_model.usd": "cattle"
    }
}

COLOR_DICT_20 = {
    "Red": [1.0, 0.0, 0.0],
    "Green": [0.0, 0.502, 0.0],
    "Blue": [0.0, 0.0, 1.0],
    "Cyan": [0.0, 1.0, 1.0],
    "Magenta": [1.0, 0.0, 1.0],
    "Yellow": [1.0, 1.0, 0.0],
    "Black": [0.0, 0.0, 0.0],
    "White": [1.0, 1.0, 1.0],
    "Gray": [0.502, 0.502, 0.502],
    "Orange": [1.0, 0.647,0.0],
    "Purple": [0.502, 0.0, 0.502],
    "Brown": [0.647, 0.165, 0.165],
    "Pink": [1.0, 0.753, 0.796],
    "Teal": [0.0, 0.502, 0.502],
    "Olive": [0.502, 0.502, 0.0],
    "Maroon": [0.502, 0.0, 0.0],
    "Gold": [1.0, 0.843, 0.0],
    "Khaki": [0.941, 0.902, 0.549],
    "RoyalBlue": [0.255, 0.412, 0.882],
    "Tomato": [1.000, 0.388, 0.278]
}


# --- Multi USD file utilities ---
def _resolve_paths(usd_path: Union[str, Sequence[str]]) -> List[str]:
    """Expand *usd_path* into a concrete list of USD file paths."""
    if isinstance(usd_path, str):
        return [usd_path] if usd_path.endswith(".usd") else OBJECT.get(usd_path, [])
    return list(usd_path)

# def _resolve_paths_categories(key: str) -> List[str]:
#     if isinstance(key, str):
#         cat_map: dict = OBJECT_CATEGORY.get(key, {})
#         paths = _resolve_paths(key) 
#         return [cat_map.get(p, "") for p in paths]

def spawn_multi_usd_file_resolved(
    prim_path: str,
    cfg: wrappers_cfg.MultiUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn multiple USD files based on the provided configurations.

    This function creates configuration instances corresponding the individual USD files and
    calls the :meth:`spawn_multi_asset` method to spawn them into the scene.

    Args:
        prim_path: The prim path to spawn the assets.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (w, x, y, z) order. Default is None.

    Returns:
        The created prim at the first prim path.
    """
    # needed here to avoid circular imports
    from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg

    usd_paths = _resolve_paths(cfg.usd_path)

    # make a template usd config
    usd_template_cfg = UsdFileCfg()
    for attr_name, attr_value in cfg.__dict__.items():
        # skip names we know are not present
        if attr_name in ["func", "usd_path", "random_choice"]:
            continue
        # set the attribute into the template
        setattr(usd_template_cfg, attr_name, attr_value)

    # create multi asset configuration of USD files
    multi_asset_cfg = MultiAssetSpawnerCfg(assets_cfg=[])
    for usd_path in usd_paths:
        usd_cfg = usd_template_cfg.replace(usd_path=usd_path)
        multi_asset_cfg.assets_cfg.append(usd_cfg)
    # set random choice
    multi_asset_cfg.random_choice = cfg.random_choice

    # propagate the contact sensor settings
    # note: the default value for activate_contact_sensors in MultiAssetSpawnerCfg is False.
    #  This ends up overwriting the usd-template-cfg's value when the `spawn_multi_asset`
    #  function is called. We hard-code the value to the usd-template-cfg's value to ensure
    #  that the contact sensor settings are propagated correctly.
    if hasattr(cfg, "activate_contact_sensors"):
        multi_asset_cfg.activate_contact_sensors = cfg.activate_contact_sensors

    # call the original function
    return spawn_multi_asset(prim_path, multi_asset_cfg, translation, orientation)

@configclass
class MultiUsdCfg(sim_utils.MultiUsdFileCfg):
    func: sim_utils.SpawnerCfg.func = spawn_multi_usd_file_resolved

    usd_path: Union[str, Sequence[str]] = MISSING
    obj_label: bool = False

    texture_path: Optional[str] = None  
    preview_surface: RandomPreviewSurfaceCfg = None

    random_texture: bool = False
    random_color: bool = False
    random_roughness: bool = False
    random_metallic: bool = False
    random_opacity: bool = False


# --- Preview surface utilities ---
@clone
def spawn_preview_surface(prim_path: str, cfg: visual_materials_cfg.PreviewSurfaceCfg) -> Usd.Prim:
    """Create a preview surface prim and override the settings with the given config.

    A preview surface is a physically-based surface that handles simple shaders while supporting
    both *specular* and *metallic* workflows. All color inputs are in linear color space (RGB).
    For more information, see the `documentation <https://openusd.org/release/spec_usdpreviewsurface.html>`__.

    The function calls the USD command `CreatePreviewSurfaceMaterialPrim`_ to create the prim.

    .. _CreatePreviewSurfaceMaterialPrim: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd.commands/omni.usd.commands.CreatePreviewSurfaceMaterialPrimCommand.html

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # spawn material if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        omni.kit.commands.execute("CreatePreviewSurfaceMaterialPrim", mtl_path=prim_path, select_new_prim=False)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    # obtain prim
    prim = prim_utils.get_prim_at_path(f"{prim_path}/Shader")
    # apply properties
    cfg = cfg.to_dict()
    del cfg["func"]
    for attr_name, attr_value in cfg.items():
        safe_set_attribute_on_usd_prim(prim, f"inputs:{attr_name}", attr_value, camel_case=True)
 
    return prim

@configclass
class RandomPreviewSurfaceCfg(sim_utils.PreviewSurfaceCfg):

    func: Callable = visual_materials.spawn_preview_surface

    diffuse_color_range: Union[Tuple[float, float, float], Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = ((0.5, 0.9), (0.5, 0.9), (0.5, 0.9))
    diffuse_color_dict: Dict[str, List[float]] = COLOR_DICT_20
    roughness_range: Union[float, Tuple[float, float]] = (0.5, 0.6)
    metallic_range: Union[float, Tuple[float, float]] = (0.5, 0.6)
    opacity_range: Union[float, Tuple[float, float]] = (0.5, 0.6)


# --- Light utilities ---
@configclass
class RandomLightCfg(AssetBaseCfg):
    prim_path="/World/light"
    spawn: sim_utils.LightCfg = None
    random_light_reset: bool = False