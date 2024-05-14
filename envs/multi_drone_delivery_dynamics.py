import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class UAVParameters:
    XY_LIM = 1.0  # default param
    XY_AXIS_RES: float
    XYDOT_LIM: float
    XYDOT_STEP: float
    PROXIMITY_THRESH: float
    CG_PROXIMITY_THRESH: float


@dataclass
class FirstOrderUAVState:
    x: float = 0.0
    y: float = 0.0


@dataclass
class FirstOrderUAVAction:
    xdot: float = 0.0
    ydot: float = 0.0


@dataclass
class FirstOrderUAVDynamics:
    timestep: float
    noise: Tuple[float, float]  # Calling random should return a 2-length tuple
    params: UAVParameters


# Will be called by MDP etc.
def get_uav_control_actions(dynamics: FirstOrderUAVDynamics) -> List[FirstOrderUAVAction]:
    fo_actions = []
    vel_vals = [val for val in
                np.arange(-dynamics.params.XYDOT_LIM, dynamics.params.XYDOT_LIM + 0.01, dynamics.params.XYDOT_STEP)]

    for xdot in vel_vals:
        for ydot in vel_vals:
            fo_actions.append(FirstOrderUAVAction(xdot, ydot))

    return fo_actions


# Samples independently
def generate_start_state(dynamics):
    x_dist = (-dynamics.params.XY_LIM, dynamics.params.XY_LIM)
    y_dist = (-dynamics.params.XY_LIM, dynamics.params.XY_LIM)

    x = random.uniform(*x_dist)
    y = random.uniform(*y_dist)

    return FirstOrderUAVState(x, y)


def get_relative_state_to_goal(goal_pos, state):
    return FirstOrderUAVState(state.x - goal_pos[0], state.y - goal_pos[1])


# Per-drone dynamics (when needed)
def next_uav_state(dynamics, state, action):
    # noiseval = random.uniform(dynamics.noise[0][0], dynamics.noise[0][1])

    xp = np.clip(state.x + action.xdot * dynamics.timestep, -dynamics.params.XY_LIM, dynamics.params.XY_LIM)
    yp = np.clip(state.y + action.ydot * dynamics.timestep, -dynamics.params.XY_LIM, dynamics.params.XY_LIM)

    return FirstOrderUAVState(xp, yp)


# Dynamics cost for 1st-order is just the velocity cost
# Will be scaled by higher-level reward function appropriately
def dynamics_cost(dynamics, a):
    return a.xdot ** 2 + a.ydot ** 2
