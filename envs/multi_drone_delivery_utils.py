## Utils to parameterically generate a problem
# NOTE: Everything below assumes XY_LIM = 1.0
import numpy as np
import random


# Computes a goal region radius appropriate for its capacity and resolution
# Basically we want it such that the inscribed square in the circle has enough
# slots to accommodate the capacity and then some. We don't want a tiling puzzle
# to have to be solved.
def get_goal_radius(capacity, axis_res):
    rp = np.sqrt((capacity * axis_res ** 2) / 2.0)
    r = (np.ceil(2 * rp / axis_res) + 1.0) * axis_res / 2.0

    return min(r, 0.5)

class CircularGoalRegion:
    def __init__(self, cen, rad, cap):
        self.cen = cen
        self.rad = rad
        self.cap = cap


# Hardcode the goal regions at the centre of each quadrant
# The capacity is randomly sampled by dividing the agents into four chunks
# and sampling around them.
def get_quadrant_goal_regions(nagents, axis_res):
    assert nagents % 8 == 0, f"Num. agents {nagents} is not a multiple of 8!"

    half_nagents = nagents // 2
    oct_nagents = nagents // 8

    cap_quad1 = random.randint(oct_nagents, 3 * oct_nagents)
    cap_quad2 = random.randint(oct_nagents, 3 * oct_nagents)

    cap_quad3 = half_nagents - cap_quad1
    cap_quad4 = half_nagents - cap_quad2

    reg_quad1 = CircularGoalRegion(cen=[0.5, 0.5], rad=get_goal_radius(cap_quad1, axis_res), cap=cap_quad1)
    reg_quad2 = CircularGoalRegion(cen=[-0.5, 0.5], rad=get_goal_radius(cap_quad2, axis_res), cap=cap_quad2)
    reg_quad3 = CircularGoalRegion(cen=[-0.5, -0.5], rad=get_goal_radius(cap_quad3, axis_res), cap=cap_quad3)
    reg_quad4 = CircularGoalRegion(cen=[0.5, -0.5], rad=get_goal_radius(cap_quad4, axis_res), cap=cap_quad4)

    goal_regs = [reg_quad1, reg_quad2, reg_quad3, reg_quad4]
    reg_to_uavid = []

    idx = 0 # in Julia it starts from 1
    for gr in goal_regs:
        reg_to_uavid.append(set(range(idx, idx + gr.cap)))
        idx += gr.cap

    return goal_regs, reg_to_uavid