from __future__ import annotations
import re
import escnn
import escnn.group
import numpy as np
from escnn.group import CyclicGroup, DihedralGroup, DirectProductGroup, Group, Representation
from omegaconf import DictConfig
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens


def get_escnn_group(cfg: DictConfig):
    """Get the ESCNN group object from the group label in the config file."""
    group_label = cfg.group_label
    label_pattern = r'([A-Za-z]+)(\d+)'
    assert cfg.group_label is not None, f'Group label unspecified. Not clear which symmetry group {cfg.name} has'
    match = re.match(label_pattern, group_label)
    if match:
        group_class = match.group(1)
        order = int(match.group(2))
    else:
        raise AttributeError(f'Group label {group_label} is not a known group label (Dn: Dihedral, Cn: Cyclic) order n')

    group_axis = np.array([0, 0, 1])
    subgroup_id = np.zeros_like(group_axis, dtype=bool).astype(object)
    if group_class.lower() == 'd':  # Dihedral
        # Define the symmetry space using presets from ESCNN
        # subgroup_id[group_axis == 1] = order
        symmetry_space = escnn.gspaces.dihedralOnR3(n=order // 2, axis=0.0)
    elif group_class.lower() == 'c':  # Cyclic
        assert order >= 2, f'Order of cyclic group must be greater than 2, got {order}'
        subgroup_id[group_axis == 1] = order
        symmetry_space = escnn.gspaces.GSpace3D(tuple(subgroup_id))
    elif group_class.lower() == "k":  # Klein four group
        is_planar_subgroup = True
        symmetry_space = escnn.gspaces.GSpace3D(sg_id=(is_planar_subgroup, False, 2))
    elif group_class.lower() == "dh":  # Dn x C2. Group of regular rectangular cuboid
        symmetry_space = escnn.gspaces.GSpace3D(sg_id=(True, True, order))
    else:
        raise NotImplementedError(f"We have to implement the case of {group_label}")

    return symmetry_space


def generate_euclidean_space_representations(G: Group) -> tuple[Representation, ...]:
    """Generate the E3 representation of the group G.

    This representation is used to transform all members of the Euclidean Space in 3D.
    I.e., points, vectors, pseudo-vectors, etc.
    TODO: List representations generated.

    Args:
        G (Group): Symmetry group of the robot.

    Returns:
        rep_E3 (Representation): Representation of the group G on the Euclidean Space in 3D.
    """
    # Configure E3 representations and group
    if isinstance(G, CyclicGroup):
        if G.order() == 2:  # Reflection symmetry
            rep_R3 = G.irrep(0) + G.irrep(1) + G.trivial_representation
        else:
            rep_R3 = G.irrep(1) + G.trivial_representation
    elif isinstance(G, DihedralGroup):
        rep_R3 = G.irrep(0, 1) + G.irrep(1, 1) + G.trivial_representation
    elif isinstance(G, DirectProductGroup):
        if G.name == "Klein4":
            rep_R3 = G.representations['rectangle'] + G.trivial_representation
            rep_R3 = escnn.group.change_basis(rep_R3, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), name="E3")
        elif G.name == "FullCylindricalDiscrete":
            rep_hx = np.array(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            rep_hy = np.array(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))
            rep_hz = np.array(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
            rep_E3_gens = {h: rep_h for h, rep_h in zip(G.generators, [rep_hy, rep_hx, rep_hz])}
            rep_E3_gens[G.identity] = np.eye(3)
            rep_R3 = group_rep_from_gens(G, rep_E3_gens)
        else:
            raise NotImplementedError(f"Direct product {G} not implemented yet.")
    else:
        raise NotImplementedError(f"Group {G} not implemented yet.")

    # Representation of unitary/orthogonal transformations in d dimensions.
    rep_R3.name = "R3"

    # We include some utility symmetry representations for different geometric objects.
    # We define a Ed as a (d+1)x(d+1) matrix representing a homogenous transformation matrix in d dimensions.
    rep_E3 = rep_R3 + G.trivial_representation
    rep_E3.name = "E3"

    # Build a representation of orthogonal transformations of pseudo-vectors.
    # That is if det(rep_O3(h)) == -1 [improper rotation] then we have to change the sign of the pseudo-vector.
    # See: https://en.wikipedia.org/wiki/Pseudovector
    pseudo_gens = {h: -1 * rep_R3(h) if np.linalg.det(rep_R3(h)) < 0 else rep_R3(h) for h in G.generators}
    pseudo_gens[G.identity] = np.eye(3)
    rep_R3pseudo = group_rep_from_gens(G, pseudo_gens)
    rep_R3pseudo.name = "R3_pseudo"

    rep_E3pseudo = rep_R3pseudo + G.trivial_representation
    rep_E3pseudo.name = "E3_pseudo"

    return rep_R3, rep_E3, rep_R3pseudo, rep_E3pseudo