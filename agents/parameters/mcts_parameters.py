from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union


@dataclass(slots=True)
class MctsParameters:
    gamma: float
    C: Union[float, int]
    action_selection_fn: Callable
    rollout_selection_fn: Callable
    max_depth: int
    n_states: int
    n_actions: int
    env: Any
    initial_state: Any
    n_sim: int
    message_passing_it: int
    number_of_agents: int
    coordination_graph: Any
    matrix: bool
    dynamic_coordination_graph: bool

@dataclass(slots=True)
class MctsSPIBBParameters:
    pi_b: Any
    MLE_T: Any
    mask: Any
    non_boot_action: Any
    hist_coord_graph: Any



