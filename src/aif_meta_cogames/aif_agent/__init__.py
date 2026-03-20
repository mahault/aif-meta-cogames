"""Discrete active inference agent for CogsGuard.

Provides a 216-state POMDP generative model (phase x hand x target_mode x role),
13 task-level policies as actions, 6 observation modalities, observation
discretizer, and tools for fitting model parameters from trajectory data.
"""

from .discretizer import (
    Hand,
    NUM_HANDS,
    NUM_OBS,
    NUM_PHASES,
    NUM_ROLES,
    NUM_STATES,
    NUM_TARGET_MODES,
    NUM_TASK_POLICIES,
    ObsContest,
    ObsInventory,
    ObsResource,
    ObsRoleSignal,
    ObsSocial,
    ObsStation,
    ObservationDiscretizer,
    Phase,
    Role,
    TargetMode,
    TaskPolicy,
    TASK_POLICY_NAMES,
    infer_task_policy,
    state_factors,
    state_index,
    state_label,
)
from .generative_model import CogsGuardPOMDP, NUM_ACTIONS, ACTION_NAMES

# Cogames policy integration (requires mettagrid/cogames installed)
try:
    from .cogames_policy import AIFBeliefState, AIFCogPolicyImpl, AIFPolicy
except ImportError:
    pass  # mettagrid not available (e.g. Windows, offline fitting)

__all__ = [
    "AIFBeliefState",
    "AIFCogPolicyImpl",
    "AIFPolicy",
    "ACTION_NAMES",
    "CogsGuardPOMDP",
    "Hand",
    "NUM_ACTIONS",
    "NUM_HANDS",
    "NUM_OBS",
    "NUM_PHASES",
    "NUM_ROLES",
    "NUM_STATES",
    "NUM_TARGET_MODES",
    "NUM_TASK_POLICIES",
    "ObsContest",
    "ObsInventory",
    "ObsResource",
    "ObsRoleSignal",
    "ObsSocial",
    "ObsStation",
    "ObservationDiscretizer",
    "Phase",
    "Role",
    "TargetMode",
    "TaskPolicy",
    "TASK_POLICY_NAMES",
    "infer_task_policy",
    "state_factors",
    "state_index",
    "state_label",
]
