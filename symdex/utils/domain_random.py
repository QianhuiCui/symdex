import torch
import omegaconf

class DomainRandomizer:
    """Handles Gaussian-to-Uniform domain randomization for Isaac environments."""

    def __init__(self, cfg, num_envs=4096):
        """
        Args:
        - cfg: dict, keys are parameter names, values are (min, max) tuples.
        - sigma_step: float, step size for increasing sigma.
        - num_envs: int, number of parallel environments in Isaac Gym.
        """
        self.cfg = cfg
        self.eval = cfg.eval
        self.total_step = cfg.total_step
        self.success_threshold = cfg.success_threshold
        self.num_envs = num_envs
        self.best_so_far = 0
        if self.eval:
            self.best_so_far = self.total_step + 1
        self.revert = False
        self.randomization_state = {}
        self.curriculum_state = {}

        # Initialize parameters
        if self.cfg.curriculum is not None:
            for param in self.cfg.curriculum.keys():
                self.curriculum_state[param] = self.cfg.curriculum[param]

        if self.cfg.randomization is not None:
            for param in self.cfg.randomization.keys():
                a, b = self.cfg.randomization[param]
                mu = (a + b) / 2
                self.randomization_state[param] = {
                    "mu": mu,
                    "sigma": 0.0,  # Start with small std deviation
                    "min_sigma": 0.0,
                    "max_sigma": (b - a) / 2,  # When sigma reaches this, switch to uniform,
                    "use_uniform": False
                }
                self.randomization_state[param]["sigma_step"] = (self.randomization_state[param]["max_sigma"] - self.randomization_state[param]["min_sigma"]) / self.total_step

    def sample(self):
        """Samples randomized values for all environments in batch."""
        randomized_values = {}
        for param, value in self.curriculum_state.items():
            if isinstance(value, omegaconf.dictconfig.DictConfig):
                value = value["values"]
            idx = min(self.best_so_far, len(value) - 1)
            randomized_values[param] = value[idx]

        for param, state in self.randomization_state.items():
            a, b = self.cfg.randomization[param]
            if state["use_uniform"]:
                randomized_values[param] = (b - a) * torch.rand(self.num_envs, device="cpu") + a
            else:
                sampled_values = torch.normal(mean=state["mu"], std=state["sigma"], size=(self.num_envs,), device="cpu") 
                sampled_values = torch.clamp(sampled_values, min=a, max=b)  # Ensure values stay within range
                randomized_values[param] = sampled_values
        return randomized_values, self.randomization_state, self.curriculum_state

    def update(self, success_rate):
        """Updates sigma and transitions to uniform distribution when necessary."""
        if success_rate > self.success_threshold:
            self.best_so_far = min(self.best_so_far + 1, self.total_step)
            self.revert = False
            for param, state in self.randomization_state.items():
                if not state["use_uniform"]:
                    state["sigma"] = state["min_sigma"] + self.best_so_far * state["sigma_step"]
                    if state["sigma"] >= state["max_sigma"]:
                        state["use_uniform"] = True  # Switch to uniform distribution
        else:
            self.revert = True
            for param, state in self.randomization_state.items():
                if state["use_uniform"]:
                    state["sigma"] = state["min_sigma"] + (self.best_so_far - 1) * state["sigma_step"]
                state["use_uniform"] = False  # Switch back to Gaussian distribution
