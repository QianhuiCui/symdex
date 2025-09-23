import torch
import numpy as np
import escnn
from morpho_symm.utils.mysc import ConfigException
from morpho_symm.utils.rep_theory_utils import escnn_representation_form_mapping, group_rep_from_gens
from morpho_symm.utils.algebra_utils import gen_permutation_matrix
from symdex.utils.symmetry_utils import generate_euclidean_space_representations, get_escnn_group


def load_symmetric_system(cfg) -> escnn.group.Group:
    """Utility function to get the symmetry group and representations of a robotic system defined in config.

    This function generates the symmetry group representations for the following
    spaces:
        1. The joint-space (Q_js), known as the space of generalized position coordinates.
        2. The joint-space tangent space (TqQ_js), known as the space of generalized velocities.
        3. The Euclidean space (Ed) in which the dynamical system evolves in.

    Args:
        cfg (DictConfig): configuration parameters of the robot. Check `cfg/`

    Returns:
        G (escnn.group.Group): Instance of the symmetry Group of the robot. The representations for Q_js, TqQ_js and
        Ed are added to the list of representations of the group.
    """
    symmetry_space = get_escnn_group(cfg)

    G = symmetry_space.fibergroup

    # Select the field for the representations.
    rep_field = float if cfg.rep_fields.lower() != 'complex' else complex

    # Assert the required representations are present in the robot configuration.
    if 'permutation_Q_js' not in cfg:
        raise ConfigException(f"Configuration file must define the field `permutation_Q_js`, "
                              f"describing the joint space permutation per each non-trivial group's generator.")
    if 'permutation_TqQ_js' not in cfg:
        raise ConfigException(f"Configuration file must define the field `permutation_TqQ_js`, "
                              f"describing the tangent joint-space permutation per each non-trivial group's generator.")

    reps_in_cfg = [k.split('permutation_')[1] for k in cfg if "permutation" in k]

    for rep_name in reps_in_cfg:
        try:
            perm_list = list(cfg[f'permutation_{rep_name}'])
            rep_dim = len(perm_list[0])
            reflex_list = list(cfg[f'reflection_{rep_name}'])
            assert len(perm_list) == len(reflex_list), \
                f"Found different number of permutations and reflections for {rep_name}"
            assert len(perm_list) >= len(G.generators), \
                f"Found {len(perm_list)} element reps for {rep_name}, Expected {len(G.generators)} generators for {G}"
            # Generate ESCNN representation of generators
            gen_rep = {}
            for h, perm, refx in zip(G.generators, perm_list, reflex_list):
                assert len(perm) == len(refx) == rep_dim
                refx = np.array(refx, dtype=rep_field)
                gen_rep[h] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)
            # Generate the entire group
            rep = group_rep_from_gens(G, rep_H=gen_rep)
            rep.name = rep_name
            G.representations.update({rep_name: rep})
        except Exception as e:
            raise ConfigException(f"Error in the definition of the representation {rep_name}") from e

    rep_Q_js = G.representations['Q_js']
    rep_TqQ_js = G.representations.get('TqQ_js', None)
    rep_TqQ_js = rep_Q_js if rep_TqQ_js is None else rep_TqQ_js

    # Create the representation of isometries on the Euclidean Space in d dimensions.
    # This adds `O3` and `E3` representations to the group.
    rep_R3, rep_E3, rep_R3pseudo, rep_E3pseudo = generate_euclidean_space_representations(G)

    # Define the representation of the rotation matrix R that transforms the base orientation.
    rep_rot_flat = {}
    for h in G.elements:
        rep_rot_flat[h] = np.kron(rep_R3(h), rep_R3(~h).T)
    rep_rot_flat = escnn_representation_form_mapping(G, rep_rot_flat)
    rep_rot_flat.name = "SO3_flat"

    # Add representations to the group.
    G.representations.update(Q_js=rep_Q_js,
                             TqQ_js=rep_TqQ_js,
                             R3=rep_R3,
                             E3=rep_E3,
                             R3_pseudo=rep_R3pseudo,
                             E3_pseudo=rep_E3pseudo,
                             SO3_flat=rep_rot_flat)

    print(f"Symmetry group has the following representations:")
    for name, rep in G.representations.items():
        print(f"\t {name}: dimension: {rep.size}")

    return G


class SymmetryManager:
    def __init__(self, cfg, symmetric_envs=False):
        self.cfg = cfg
        self.symmetric_envs = symmetric_envs
        self.obs_dim = cfg.single_agent_obs_dim
        self.obs_idx = cfg.single_agent_obs_idx
        if self.symmetric_envs:
            self.obs_idx_symmetry = cfg.single_agent_obs_idx_symmetry
        self.action_dim = cfg.single_agent_action_dim

    def get_multi_agent_obs(self, obs, env_symmetry_idx=None):
        """
        Get the observation for each policy
        env_symmetry_idx: a tensor that indicates if the environment is in symmetry mode
        """
        if not self.symmetric_envs:
            return [slice_tensor(obs, self.obs_idx[i]).clone() for i in range(len(self.obs_idx))]
        
        original_idx = torch.where(env_symmetry_idx == 0)[0]
        symmetry_idx = torch.where(env_symmetry_idx == 1)[0]
        assert len(original_idx) + len(symmetry_idx) == len(obs), "the sum of original and symmetry environments should equal to the number of environments"
        
        multi_agent_obs = []
        for i in range(len(self.obs_dim)):
            # initialize the observations
            cur_obs = torch.zeros(len(obs), self.obs_dim[i], device=obs.device)
            # fill in the original observations
            cur_obs[original_idx] = slice_tensor(obs[original_idx], self.obs_idx[i])
            # fill in the symmetry observations
            cur_obs[symmetry_idx] = slice_tensor(obs[symmetry_idx], self.obs_idx_symmetry[i])
            multi_agent_obs.append(cur_obs)
        
        return multi_agent_obs
    
    def get_execute_action(self, action1, action2, env_symmetry_idx=None):
        """
        Get the action for the environment, namely combine the actions of right and left arms
        """
        if not self.symmetric_envs or not self.cfg.swap_action:
            return torch.cat([action1, action2], dim=-1)
        original_idx = torch.where(env_symmetry_idx == 0)[0]
        symmetry_idx = torch.where(env_symmetry_idx == 1)[0]
        assert len(original_idx) + len(symmetry_idx) == len(env_symmetry_idx), "the sum of original and symmetry environments should equal to the number of environments"
        assert len(action1) == len(action2) == len(env_symmetry_idx), "the number of actions and environments should match"
        assert action1.shape[-1] == action2.shape[-1] == self.action_dim, "the dimension of actions should match"
        
        action = torch.zeros(len(env_symmetry_idx), action1.shape[-1] + action2.shape[-1], device=action1.device)
        action[original_idx] = torch.cat([action1[original_idx], action2[original_idx]], dim=-1)
        action[symmetry_idx] = torch.cat([action2[symmetry_idx], action1[symmetry_idx]], dim=-1)
        return action

    def get_multi_agent_rew(self, detailed_reward, env_symmetry_idx=None):
        """
        Get the reward for each policy
        """
        single_agent_rew = self.cfg.single_agent_rew
        single_agent_rew_symmetry = self.cfg.single_agent_rew_symmetry if self.symmetric_envs else None
            
        if not self.symmetric_envs:
            rewards = parse_multi_rew(detailed_reward, single_agent_rew)
        else:
            original_idx = torch.where(env_symmetry_idx == 0)[0]
            symmetry_idx = torch.where(env_symmetry_idx == 1)[0]
            assert len(original_idx) + len(symmetry_idx) == len(env_symmetry_idx), "the sum of original and symmetry environments should equal to the number of environments"
            rewards_original = parse_multi_rew(detailed_reward, single_agent_rew, target_idx=original_idx)
            rewards_symmetry = parse_multi_rew(detailed_reward, single_agent_rew_symmetry, target_idx=symmetry_idx)
            assert len(rewards_original) == len(rewards_symmetry), "the number of rewards should match"
            rewards = [torch.zeros(len(env_symmetry_idx), device=env_symmetry_idx.device) for _ in range(len(rewards_original))]
            for i in range(len(rewards_original)):
                rewards[i][original_idx] = rewards_original[i]
                rewards[i][symmetry_idx] = rewards_symmetry[i]

        return rewards

# parse multi-agent reward
def parse_multi_rew(rew_dict, rew_list, target_idx=None):
    if target_idx is None:
        target_idx = slice(None)
    assert rew_list is not None, "single_agent_rew is not defined in the cfg"
    multi_rew = []
    for i in range(len(rew_list)):
        rew_terms = rew_list[i]
        tot_rew = None
        for rew_term, rew in rew_dict.items():
            if rew_term in rew_terms:
                if tot_rew is None:
                    tot_rew = rew[target_idx].clone()
                else:
                    tot_rew += rew[target_idx].clone()
        multi_rew.append(tot_rew)
    return multi_rew

def slice_tensor(tensor, indices):
    # If there's only one range, no need to concatenate
    if len(indices) == 1:
        start, end = indices[0]
        return tensor[:, start:end]
    else:
        # For multiple ranges, concatenate the slices along the second dimension
        slices = [tensor[:, elements[0]:elements[1]] for elements in indices]
        return torch.cat(slices, dim=1)