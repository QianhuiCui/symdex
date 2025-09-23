import ast
import platform
import random
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf, open_dict
from isaaclab_tasks.utils import load_cfg_from_registry

def set_random_seed(seed=None):
    if seed is None:
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        seed = random.randint(min_seed_value, max_seed_value)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info(f'Setting random seed to:{seed}')
    return seed

def init_wandb(cfg):
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True,
                                       throw_on_missing=True)
    wandb_cfg['hostname'] = platform.node()
    wandb_kwargs = cfg.logging.wandb
    wandb_tags = wandb_kwargs.get('tags', None)
    if wandb_tags is not None and isinstance(wandb_tags, str):
        wandb_kwargs['tags'] = [wandb_tags]
    # if cfg.artifact is not None:
    #     wandb_id = cfg.artifact.split("/")[-1].split(":")[0]
    #     wandb_run = wandb.init(**wandb_kwargs, config=wandb_cfg, id=wandb_id, resume="must")
    # else:
    wandb_run = wandb.init(**wandb_kwargs, config=wandb_cfg)
    logger.warning(f'Wandb run dir:{wandb_run.dir}')
    logger.warning(f'Project name:{wandb_run.project_name()}')
    return wandb_run

def load_class_from_path(cls_name, path):
    mod_name = 'MOD%s' % cls_name
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)

def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of the filename:{file_name}')
    return file_name

def list_class_names(dir_path):
    """
    Return the mapping of class names in all files
    in dir_path to their file path.
    Args:
        dir_path (str): absolute path of the folder.
    Returns:
        dict: mapping from the class names in all python files in the
        folder to their file path.
    """
    dir_path = pathlib_file(dir_path)
    py_files = list(dir_path.rglob('*.py'))
    py_files = [f for f in py_files if f.is_file() and f.name != '__init__.py']
    cls_name_to_path = dict()
    for py_file in py_files:
        with py_file.open(encoding="utf-8") as f:
            node = ast.parse(f.read())
        classes_in_file = [n for n in node.body if isinstance(n, ast.ClassDef)]
        cls_names_in_file = [c.name for c in classes_in_file]
        for cls_name in cls_names_in_file:
            cls_name_to_path[cls_name] = py_file
    return cls_name_to_path

def capture_keyboard_interrupt():
    import signal
    import sys
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

def preprocess_cfg(cfg):
    with open_dict(cfg):
        cfg.available_gpus = torch.cuda.device_count()
    cfg.env_name = cfg.task.env_name

    env_cfg = load_cfg_from_registry(cfg.env_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.max_episode_length * env_cfg.decimation * env_cfg.sim.dt

    # update rew term
    rew_term = env_cfg.rewards.to_dict()
    for key, value in cfg.task.rew.items():
        if key in rew_term:
            rew_term[key]['weight'] = float(value)
    env_cfg.rewards.from_dict(rew_term)

    # add hydra config to env_cfg
    env_cfg.hydra_cfg = cfg
    env_cfg.seed = cfg.seed

    # update entropy scale for each task
    task_name = cfg.task.env_name
    task_entropy_scale = {
        "BoxLiftEnv-v0": 0.0,
        "InsertDrawerEnv-v0": 0.01,
        "HandoverEnv-v0": 0.005,
        "PickObjectEnv-v0": 0.005,
        "StirBowlEnv-v0": 0.01,
        "ThreadingEnv-v0": 0.01,
    }
    # only change the scale if the user does not pass in a new scale (default is 1.0)
    if task_name in task_entropy_scale:
        cfg.algo.lambda_entropy = task_entropy_scale[task_name]

    return cfg, env_cfg

def aggregate_traj_info(infos, key, single_info=False):
    if single_info:
        infos = [infos]
    if isinstance(infos[0], Sequence):
        out = []
        for info in infos:
            time_out = []
            for env_info in info:
                time_out.append(env_info[key])
            out.append(np.stack(time_out))
        out = stack_data(out)
    elif isinstance(infos[0], dict):
        out = []
        for info in infos:
            tensor = info[key]
            out.append(tensor)
        out = stack_data(out)
    else:
        raise NotImplementedError
    if single_info:
        out = out.squeeze(0)
    return out


def stack_data(data, torch_to_numpy=False, dim=0):
    if isinstance(data[0], dict):
        out = dict()
        for key in data[0].keys():
            out[key] = stack_data([x[key] for x in data], dim=dim)
        return out
    try:
        ret = torch.stack(data, dim=dim)
        if torch_to_numpy:
            ret = ret.cpu().numpy()
    except:
        # if data is a list of arrays that do not have same shapes (such as point cloud)
        ret = data
    return ret


class Tracker:
    def __init__(self, max_len):
        self.moving_average = deque([0 for _ in range(max_len)], maxlen=max_len)
        self.max_len = max_len

    def __repr__(self):
        return self.moving_average.__repr__()

    def update(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            self.moving_average.extend(value.tolist())
        elif isinstance(value, Sequence):
            self.moving_average.extend(value)
        else:
            self.moving_average.append(value)

    def mean(self):
        return np.mean(self.moving_average)

    def std(self):
        return np.std(self.moving_average)

    def max(self):
        return np.max(self.moving_average)