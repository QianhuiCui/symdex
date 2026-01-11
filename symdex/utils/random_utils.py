from __future__ import annotations

import os
import re
import random
import torch

from pxr import Gf, Usd, UsdLux, UsdShade
import omni.kit.commands
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from symdex.utils.random_cfg import OBJECT_CATEGORY


# --- Surface randomization utilities ---
def randomize_surface(prims, color=None, roughness=None, metallic=None):
    if color is None and roughness is None and metallic is None:
        return
    for i, prim in enumerate(prims):
        mats = _iter_direct_materials(prim)
        for mat in mats:
            mat_prim = mat.GetPrim()
            for child in mat_prim.GetChildren():
                if child.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(child)

                    all_inputs = [i.GetBaseName() for i in shader.GetInputs()]

                    # randomize color
                    if color is not None:
                        for name in ("diffuse_factor", "base_color_factor", "diffuseColor", "diffuse_color"):
                            if name in all_inputs:
                                # inp = shader.GetInput(name)
                                shader.GetInput(name).Set(Gf.Vec3f(*color[i].tolist()))
                                # if inp:
                                    # inp.Set(Gf.Vec3f(*color[i].tolist()))
                                break
                        for name in ("emissive_color", "emissive_factor", "emissiveColor"):
                            if name in all_inputs:
                                shader.GetInput(name).Set(Gf.Vec3f(*color[i].tolist()))
                                break
                    # randomize roughness
                    if roughness is not None:
                        for name in ("roughness", "roughness_factor", "glossiness_factor"):
                            if name in all_inputs:
                                # inp = shader.GetInput(name)
                                # if inp:
                                if name == "glossiness_factor":
                                    shader.GetInput(name).Set(1.0 - float(roughness[i]))
                                else:
                                    shader.GetInput(name).Set(float(roughness[i]))
                                break
                    # randomize metallic
                    if metallic is not None:
                        for name in ("metallic", "metallic_factor", "specular_factor"):
                            if name in all_inputs:
                                # inp = shader.GetInput(name)
                                shader.GetInput(name).Set(float(metallic[i]))
                                # if inp:
                                    # inp.Set(float(metallic[i]))
                                break

def randomise_colors(prims, color):
    # print(f"randomize_usd_prims_colour called with {len(prims)} prims")
    for prim in prims:
        mats = _iter_direct_materials(prim)
        for mat in mats:
            # print(f"Assigning colour {color} to {mat.GetPrim().GetPath()}")
            mat_prim = mat.GetPrim()
            for child in mat_prim.GetChildren():
                if child.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(child)
                    # print("Shader type:", shader.GetIdAttr().Get())

                    all_inputs = [i.GetBaseName() for i in shader.GetInputs()]
                    # print("Available color inputs:", all_inputs)
                    
                    for name in ("diffuse_factor", "base_color_factor", "diffuseColor", "diffuse_color"):
                        if name in all_inputs:
                            inp = shader.GetInput(name)
                            if inp:
                                # print(f"Before base_color_factor: {inp.Get()}")
                                inp.Set(Gf.Vec3f(*color))
                                # print(f"After base_color_factor: {inp.Get()}")
                                break

                    # tex_inp = shader.GetInput("base_color_texture")
                    # if tex_inp and tex_inp.Get():
                    #     tex_inp.Set(None)

def randomise_roughness(prims, roughness):
    # print(f"randomize_usd_prims_roughness called with {len(prims)} prims")
    for prim in prims:
        mats = _iter_direct_materials(prim)
        for mat in mats:
            # print(f"Assigning roughness {roughness} to {mat.GetPrim().GetPath()}")
            mat_prim = mat.GetPrim()
            for child in mat_prim.GetChildren():
                if child.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(child)

                    all_inputs = [i.GetBaseName() for i in shader.GetInputs()]
                    # print("Available shader inputs:", all_inputs)
                    
                    if "glossiness_factor" in all_inputs:
                        inp = shader.GetInput("glossiness_factor")
                        if inp:
                            glossiness = 1.0 - roughness  
                            inp.Set(glossiness)

                    for name in ("roughness", "roughness_factor"):
                        inp = shader.GetInput(name)
                        if inp:
                            inp.Set(roughness)
                            break

def randomise_metallic(prims, metallic):
    # print(f"randomize_usd_prims_metallic called with {len(prims)} prims")
    for prim in prims:
        mats = _iter_direct_materials(prim)
        for mat in mats:
            # print(f"Assigning metallic {metallic} to {mat.GetPrim().GetPath()}")
            mat_prim = mat.GetPrim()
            for child in mat_prim.GetChildren():
                if child.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(child)

                    all_inputs = [i.GetBaseName() for i in shader.GetInputs()]
                    # print("Available shader inputs:", all_inputs)

                    for name in ("metallic", "metallic_factor", "specular_factor"):
                        if name in all_inputs:
                            inp = shader.GetInput(name)
                            if inp:
                                _set_shader_input(inp, metallic)
                                break

def randomise_opacity(prims, opacity):
    # print(f"randomize_usd_prims_opacity called with {len(prims)} prims")
    for prim in prims:
        mats = _iter_direct_materials(prim)
        for mat in mats:
            # print(f"Assigning opacity {opacity} to {mat.GetPrim().GetPath()}")
            mat_prim = mat.GetPrim()
            for child in mat_prim.GetChildren():
                if child.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(child)

                    all_inputs = [i.GetBaseName() for i in shader.GetInputs()]
                    # print("Available shader inputs:", all_inputs)
                    
                    for name in ("opacity", "opacity_factor", "transparency"):  # , "base_alpha", "alpha"):
                        if name in all_inputs:
                            inp = shader.GetInput(name)
                            if inp:
                                inp.Set(opacity)
                                break

def randomise_textures(prims, texture_dir):
    # print(f"randomize_usd_prims_texture called with {len(prims)} prims")
    png_files = [
        os.path.join(texture_dir, f)
        for f in os.listdir(texture_dir)
        if f.lower().endswith(".png")
    ]
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in directory: {texture_dir}")
    # print("chosen png file: ", png_files)

    for prim in prims:
        mats = _iter_direct_materials(prim)
        for mat in mats:
            mat_prim = mat.GetPrim()
            for child in mat_prim.GetChildren():
                if child.GetTypeName() == "Shader":
                    shader = UsdShade.Shader(child)
                    all_inputs = [i.GetBaseName() for i in shader.GetInputs()]
                    print("Available shader inputs:", all_inputs)

                    found = False
                    for name in ("diffuse_texture", "diffuseTexture", "base_color_texture", "albedo_texture"):
                        inp = shader.GetInput(name)
                        if inp:
                            texture_path = random.choice(png_files)
                            inp.Set(texture_path)
                            found = True
                            break
                    if not found:
                        for inp in shader.GetInputs():
                            if (inp.GetTypeName() == "asset" or inp.GetTypeName() == "asset[]") and \
                               ("texture" in inp.GetBaseName().lower() or "map" in inp.GetBaseName().lower()):
                                texture_path = random.choice(png_files)
                                inp.Set(texture_path)
                                print(f"Set {inp.GetBaseName()} for {mat_prim.GetPath()} with {texture_path}")
                                found = True
                                break
                    if not found:
                        print(f"No texture input found for shader at {mat_prim.GetPath()}, available inputs: {all_inputs}")

def _iter_direct_materials(root_prim):
    mats = {}
    for prim in Usd.PrimRange(root_prim):
        bind = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        if not bind:
            continue
        mat = bind.GetMaterial()
        if mat and mat.GetPrim() and mat.GetPrim().IsValid():
            mats[mat.GetPath()] = mat
    return list(mats.values())

def _set_shader_input(inp, value):
    type_name = inp.GetTypeName()
    if type_name in ("float", "double"):
        inp.Set(float(value))
    elif type_name in ("float3", "double3", "color3f", "color3d", "vector3f", "GfVec3f"):
        if not isinstance(value, (tuple, list)):
            value = (value, value, value)
        inp.Set(Gf.Vec3f(*value))
    else:
        inp.Set(value)


# --- Light creation utilities ---
def get_light(random_light: bool):
    if random_light:
        light_types = [
            sim_utils.DomeLightCfg,
            sim_utils.DistantLightCfg,
        ]
        LightType = random.choice(light_types)
        if LightType is sim_utils.DomeLightCfg:
            random_color = (random.uniform(0.5, 1.0), random.uniform(0.5, 1.0), random.uniform(0.5, 1.0),)
            random_intensity = random.uniform(500.0, 5000.0)
            return LightType(
                    color=random_color,
                    intensity=random_intensity,
                )
        elif LightType is sim_utils.DistantLightCfg:
            random_color = (random.uniform(0.5, 1.0), random.uniform(0.5, 1.0), random.uniform(0.5, 1.0),)
            random_intensity = random.uniform(2000.0, 3000.0)
            return LightType(
                    color=random_color,
                    intensity=random_intensity,
                )
    else:
        return sim_utils.DomeLightCfg(
                # color=(random.uniform(0.5, 1.0), random.uniform(0.5, 1.0), random.uniform(0.5, 1.0),),
                color=(0.75, 0.75, 0.75),
                # intensity=random.uniform(2500.0, 3000.0),
                intensity=2500.0,
            )

def create_light(stage, prim_path, light_cfg, translation=None):
    # prim = stage.DefinePrim(prim_path, light_cfg.prim_type)
    if light_cfg.prim_type == "CylinderLight":
        prim = UsdLux.CylinderLight.Define(stage, prim_path)
    elif light_cfg.prim_type == "DomeLight":
        prim = UsdLux.DomeLight.Define(stage, prim_path)
    elif light_cfg.prim_type == "DistantLight":
        prim = UsdLux.DistantLight.Define(stage, prim_path)
    elif light_cfg.prim_type == "SphereLight":
        prim = UsdLux.SphereLight.Define(stage, prim_path)
    elif light_cfg.prim_type == "DiskLight":
        prim = UsdLux.DiskLight.Define(stage, prim_path)
    else:
        raise ValueError(f"Unknown light type: {light_cfg.prim_type}")

    prim.GetColorAttr().Set(light_cfg.color)
    prim.GetIntensityAttr().Set(light_cfg.intensity)
    if hasattr(light_cfg, "radius"):
        prim.GetRadiusAttr().Set(light_cfg.radius)
    if light_cfg.prim_type == "SphereLight" or light_cfg.prim_type == "DiskLight" or light_cfg.prim_type == "CylinderLight":
        if translation is not None:
            x, y, z = translation
            prim.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
    return prim 


# --- Language label utilities---
LANGUAGE_TEMPLATES: dict[str, str] = {
    "default": "{color} {surface} {obj}",
    "insertDrawer": "pick {color} {surface} {obj} into drawer and insert drawer",
    "boxLift": "lift {color} {surface} tote ",
    "handover": "right-to-left handover of a {color} {surface} {obj}",
    "pickObject": "right hand pick up the {color_1} {surface_1} {obj_1}, and then left hand pick up the {color_2} {surface_2} {obj_2}",
    "stirBowl": "",
    "threading": "",
    "pouring": "",
}

SURFACE_THRESHOULDS = {
    "metallic": {
        "high": 0.8,
        "medium": 0.5,
    },
    "roughness": {
        "polished": 0.1,
        "glossy": 0.3,
        "satin": 0.6,
        "matte": 0.9,
    },
}

def get_surface_description(roughness, metallic):
    roughness = roughness.squeeze(-1) 
    metallic  = metallic.squeeze(-1)

    rough_list = roughness.tolist()
    metal_list = metallic.tolist()
    labels = []

    for r, m in zip(rough_list, metal_list):
        # descriptor by roughness
        if r <= SURFACE_THRESHOULDS["roughness"]["polished"]:
            descriptor = "polished"
        elif r <= SURFACE_THRESHOULDS["roughness"]["glossy"]:
            descriptor = "glossy"
        elif r <= SURFACE_THRESHOULDS["roughness"]["satin"]:
            descriptor = "satin"
        elif r <= SURFACE_THRESHOULDS["roughness"]["matte"]:
            descriptor = "matte"
        else:
            descriptor = "rough"

        # category by metallic
        if m >= SURFACE_THRESHOULDS["metallic"]["high"]:
            # pure metal
            label = f"{descriptor} metal"
        elif m >= SURFACE_THRESHOULDS["metallic"]["medium"]:
            # semi-metallic blend
            label = f"{descriptor} semi-metallic"
        else:
            # non-metal: plastic/ceramic
            if descriptor == "polished":
                label = "porcelain-like"
            else:
                label = f"{descriptor} plastic"
        labels.append(label)
    
    return labels

def get_lang_label(
        task: str | None, 
        num_envs: int, 
        objs: list[list[str]], 
        colors: list[list[str]], 
        surfaces: list[list[str]], 
    ) -> list[str]:
    template = LANGUAGE_TEMPLATES.get(task or "default", LANGUAGE_TEMPLATES["default"])
    
    num_obj_labels = len(objs)
    lang_labels = []

    if num_obj_labels == 0:
        for i in range(num_envs):
            text = template.format(
                color=colors[0][i],
                surface=surfaces[0][i],
            )
            lang_labels.append(" ".join(text.split()).strip())
    elif num_obj_labels == 1:
        for i in range(num_envs):
            text = template.format(
                color=colors[0][i],
                surface=surfaces[0][i],
                obj=objs[0][i],
            )
            lang_labels.append(" ".join(text.split()).strip())
    else:
        for i in range(num_envs):
            text = template.format(
                color_1=colors[0][i],
                surface_1=surfaces[0][i],
                obj_1=objs[0][i],
                color_2=colors[1][i],
                surface_2=surfaces[1][i],
                obj_2=objs[1][i],
            )
            lang_labels.append(" ".join(text.split()).strip())
    return lang_labels
    # return [
    #     f"{color} {surface} {obj}"
    #     for color, surface, obj in zip(color, surface, obj)
    # ]


# --- Random sampling utilities ---
# def find_prims_by_regex(stage, pattern_str):
#     pattern = re.compile(pattern_str)
#     return [prim for prim in Usd.PrimRange(stage.GetPseudoRoot())
#             if pattern.match(prim.GetPath().pathString)]

def sample_value(val):
    if isinstance(val, tuple) and isinstance(val[0], tuple):
        return tuple(random.uniform(a, b) for a, b in val)
    elif isinstance(val, tuple) and len(val) == 2:
        return random.uniform(val[0], val[1])
    else:
        return val
    
def sample_tuple(val, length, device):
    if isinstance(val, tuple) and isinstance(val[0], tuple):
        upper_bound = torch.tensor([b for a, b in val], device=device)
        lower_bound = torch.tensor([a for a, b in val], device=device)
        return math_utils.sample_uniform(lower_bound, upper_bound, (length, len(val)), device)
    elif isinstance(val, tuple) and len(val) == 2:
        return math_utils.sample_uniform(val[0], val[1], (length, 1), device)
    else:
        raise ValueError(f"Invalid value: {val}")
    
# def sample_list(usd_path, length, device):
#     if isinstance(usd_path, str):
#         paths = ([usd_path] if usd_path.endswith(".usd") else OBJECT.get(usd_path, []))
#     else:
#         paths = list(usd_path)
#     if not paths:
#         raise ValueError(f"No .usd paths found for: {usd_path}")
#     idx = torch.randint(0, len(paths), (length,), device=device)
#     return [paths[i] for i in idx.cpu().tolist()]

def get_object_label(usd_path, prims):
    label = []
    category_dict = OBJECT_CATEGORY[usd_path]
    for prim in prims:
        chosen_path = prim.GetCustomDataByKey("sourceUsdPath")
        label.append(category_dict[chosen_path])
    return label

def sample_dict(dict, length, device):
    label = []
    value = []
    for i in range(length):
        cur_key = random.choice(list(dict.keys()))
        label.append(cur_key.lower())
        value.append(dict[cur_key])
    return label, torch.tensor(value, device=device)