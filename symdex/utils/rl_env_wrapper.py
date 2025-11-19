from __future__ import annotations

import gym.spaces
import torch
import numpy as np
import math
import einops
from enum import IntEnum
import torchvision.transforms as T
from isaaclab.envs import VecEnvObs

"""
Vectorized environment wrapper.
"""


class VecEnvWrapper:
    def __init__(self, env, rl_device: str, clip_obs: float = np.inf, clip_actions: float = 1, raw_data: bool = False, image_size: int = 128):
        """Initializes the wrapper instance.

        Args:
            env: The environment to wrap around.
            rl_device: The device on which agent computations are performed.
            clip_obs: The clipping value for observations.
            clip_actions: The clipping value for actions.

        Raises:
            ValueError: The environment is not inherited from :class:`ManagerBasedRLEnv`.
            ValueError: If specified, the privileged observations (critic) are not of type :obj:`gym.spaces.Box`.
        """
        # initialize the wrapper
        self.env = env
        # store provided arguments
        self._rl_device = rl_device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self._sim_device = env.unwrapped.device
        self.max_episode_length_s = env.unwrapped.cfg.episode_length_s
        self.max_episode_length = math.ceil(env.unwrapped.cfg.episode_length_s / (env.unwrapped.cfg.decimation * env.unwrapped.physics_dt))
        self.raw_data = raw_data
        self.cam_enable = env.unwrapped.cfg.hydra_cfg.task.cam.enable
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        
    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return (
            f"<{type(self).__name__}{self.env}>"
            f"\n\tObservations clipping: {self._clip_obs}"
            f"\n\tActions clipping     : {self._clip_actions}"
            f"\n\tAgent device         : {self._rl_device}"
        )

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        obs_space = self.unwrapped.observation_space["policy"]
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_obs, self._clip_obs, obs_space.shape)

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        action_space = self.unwrapped.action_space
        action_space_shape = action_space.shape
        # return casted space in gym.spaces.Box (OpenAI Gym)
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space_shape)

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def num_envs(self) -> int:
        """Returns the number of sub-environment instances."""
        return self.unwrapped.num_envs

    @property
    def device(self) -> str:
        """Returns the base environment simulation device."""
        return self.unwrapped.device

    def get_number_of_agents(self) -> int:
        """Returns number of actors in the environment."""
        return getattr(self, "num_agents", 1)

    def get_env_info(self) -> dict:
        """Returns the Gym spaces for the environment."""
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self):  # noqa: D102
        obs_dict, extras = self.env.reset()
        # process observations and states
        return self._process_obs(obs_dict), extras

    def step(self, actions):  # noqa: D102
        actions = actions.detach().clone().to(device=self._sim_device)
        # clip the actions
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        # perform environment step
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        # note: only useful when `value_bootstrap` is True in the agent configuration
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)
        # process observations and states
        obs = self._process_obs(obs_dict)
        # move buffers to rl-device
        # note: we perform clone to prevent issues when rl-device and sim-device are the same.
        rew = rew.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)
        extras = {
            k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
        }
        # remap extras from "log" to "episode"
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        return obs, rew, dones.float(), extras

    def close(self):  # noqa: D102
        return self.env.close()

    # """
    # Helper functions
    # """

    # def _process_obs(self, obs_dict: VecEnvObs) -> torch.Tensor | dict[str, torch.Tensor]:
    #     """Processing of the observations and states from the environment.

    #     Note:
    #         States typically refers to privileged observations for the critic function. It is typically used in
    #         asymmetric actor-critic algorithms.

    #     Args:
    #         obs_dict: The current observations from environment.

    #     Returns:
    #         If environment provides states, then a dictionary containing the observations and states is returned.
    #         Otherwise just the observations tensor is returned.
    #     """
    #     # process policy obs
    #     obs = obs_dict["policy"]
    #     # clip the observations
    #     obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)
    #     # move the buffer to rl-device
    #     obs = obs.to(device=self._rl_device).clone()
        
    #     return obs

    # """
    # Helper functions
    # """

    def _process_obs(self, obs_dict: VecEnvObs) -> torch.Tensor | dict[str, torch.Tensor]:
        """Processing of the observations and states from the environment.

        Note:
            States typically refers to privileged observations for the critic function. It is typically used in
            asymmetric actor-critic algorithms.

        Args:
            obs_dict: The current observations from environment.

        Returns:
            If environment provides states, then a dictionary containing the observations and states is returned.
            Otherwise just the observations tensor is returned.
        """
        if self.cam_enable:
            for key, value in obs_dict.items():
                if key == "critic" or "policy" in key:
                    obs_dict[key] = torch.clamp(value, -self._clip_obs, self._clip_obs).to(device=self._rl_device).clone()
                elif "vision" in key:
                    if not self.raw_data:
                        if obs_dict[key].shape[1] == 2:
                            agentview_img = self._process_image_sequence(value[:, 0]).unsqueeze(1)
                            eye_in_hand_img = self._process_image_sequence(value[:, 1]).unsqueeze(1)
                            imgs = torch.cat([agentview_img, eye_in_hand_img], dim=1)
                            obs_dict[key] = imgs
                    else:
                        obs_dict[key] = value.clone().to(device=self._rl_device)
                elif "point_cloud" in key or "depth" in key:
                    obs_dict[key] = value.clone().to(device=self._rl_device)
            
            return obs_dict
        else:
            obs = obs_dict["policy"]
            obs = torch.clamp(obs, -self._clip_obs, self._clip_obs).to(device=self._rl_device).clone()
            return obs
        
    def _process_image_sequence(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Input: np_imgs of shape (T, H, W, C), dtype=uint8
        Output: Tensor of shape (T, C, H, W), float32, transformed
        """
        imgs = imgs.float() / 255.0
        imgs = einops.rearrange(imgs, "T H W C -> T C H W")
        return torch.stack([self.image_transform(img) for img in imgs], dim=0)

# ------------------------------------------------------
# Adapter from Gym to DM Env
# ------------------------------------------------------
class StepType(IntEnum):
    FIRST = 0
    MID   = 1
    LAST  = 2

class TimestepCompat:
    def __init__(self, step_type, reward, discount, observation):
        self.step_type   = step_type
        self.reward      = reward
        self.discount    = discount
        self.observation = observation

    def first(self): return self.step_type == StepType.FIRST
    def mid(self):   return self.step_type == StepType.MID
    def last(self):  return self.step_type == StepType.LAST

# ---- utils ----
@staticmethod
def _to_numpy(x):
    import numpy as np
    import torch
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k:_to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_numpy(v) for v in x]
    return x  

class GymToDMEnv:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        obs = _to_numpy(obs)
        reward   = np.array(0.0, dtype=np.float32)   # reset 步奖励置 0
        discount = np.array(1.0, dtype=np.float32)   # reset 步折扣置 1
        return TimestepCompat(StepType.FIRST, reward, discount, obs)

    def step(self, action):
        out = self._env.step(action)
        # Gymnasium: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, _info = out
            done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        # Gym: (obs, reward, done, info)
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, _info = out
            done = bool(np.asarray(done).any())
        else:
            raise RuntimeError(f"Unexpected step() output: {type(out)} with value {out}")

        obs = _to_numpy(obs)
        reward = _to_numpy(reward)
        discount = np.array(0.0 if done else 1.0, dtype=np.float32)
        step_type = StepType.LAST if done else StepType.MID
        return TimestepCompat(step_type, reward, discount, obs)

    def observation_spec(self):
        return {}
    def action_spec(self):
        return {}
    def reward_spec(self):
        return {}
    def discount_spec(self):
        return {}