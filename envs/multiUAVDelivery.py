import random
import numpy as np
import networkx as nx
from dataclasses import dataclass
from itertools import product
from envs.multi_drone_delivery_dynamics import FirstOrderUAVAction, get_uav_control_actions, UAVParameters, \
    FirstOrderUAVDynamics, next_uav_state, FirstOrderUAVState, dynamics_cost, generate_start_state
from typing import NamedTuple, Tuple
from envs.multi_drone_delivery_utils import get_quadrant_goal_regions
from collections import namedtuple
import matplotlib.pyplot as plt


## Each UAV's action is a dynamics action or a boarding action.
@dataclass
class UAVGeneralAction:
    dyn_action: FirstOrderUAVAction
    to_board: bool
    no_op: bool


GridCoords = namedtuple('GridCoords', ['x', 'y'])
int_to_act = {
    0: UAVGeneralAction(FirstOrderUAVAction(xdot=-0.2, ydot=-0.2), False, False),
    1: UAVGeneralAction(FirstOrderUAVAction(xdot=-0.2, ydot=0.0), False, False),
    2: UAVGeneralAction(FirstOrderUAVAction(xdot=-0.2, ydot=0.2), False, False),
    3: UAVGeneralAction(FirstOrderUAVAction(xdot=0.0, ydot=-0.2), False, False),
    4: UAVGeneralAction(FirstOrderUAVAction(xdot=0.0, ydot=0.2), False, False),
    5: UAVGeneralAction(FirstOrderUAVAction(xdot=0.2, ydot=-0.2), False, False),
    6: UAVGeneralAction(FirstOrderUAVAction(xdot=0.2, ydot=0.0), False, False),
    7: UAVGeneralAction(FirstOrderUAVAction(xdot=0.2, ydot=0.2), False, False),
    8: UAVGeneralAction(FirstOrderUAVAction(xdot=0.0, ydot=0.0), True, False),
    9: UAVGeneralAction(FirstOrderUAVAction(xdot=0.0, ydot=0.0), False, True)
}

act_to_int = {
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=-0.2, ydot=-0.2), to_board=False, no_op=False)': 0,
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=-0.2, ydot=0.0), to_board=False, no_op=False)': 1,
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=-0.2, ydot=0.2), to_board=False, no_op=False)': 2,
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.0, ydot=-0.2), to_board=False, no_op=False)': 3,
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.0, ydot=0.2), to_board=False, no_op=False)': 4,
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.2, ydot=-0.2), to_board=False, no_op=False)': 5,
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.2, ydot=0.0), to_board=False, no_op=False)': 6,
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.2, ydot=0.2), to_board=False, no_op=False)': 7,
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.0, ydot=0.0), to_board=True, no_op=False)': 8,
    'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.0, ydot=0.0), to_board=False, no_op=True)': 9
}


## Helper functions and structs for defining the environment
# The subregions that drones have to reach - centre, radius, capacity
class CircularGoalRegion(NamedTuple):
    cen: Tuple[float, float]
    rad: float
    cap: int


def is_in_region(reg, gc):
    distance = np.sqrt((reg.cen[0] - gc.x) ** 2 + (reg.cen[1] - gc.y) ** 2)
    return distance <= reg.rad


@dataclass
class UAVGeneralState:
    coords: GridCoords[float, float]
    boarded: bool


# Templated on state-action-dynamics types
class MultiUAVDeliveryMDPConstructor:
    def __init__(self, n_agents, grid_side, dynamics, per_uav_actions, discount, goal_regions,
                 region_to_uavids, uav_to_region_map, constant_cg_adj_mat, reach_goal_bonus,
                 proximity_penalty_scaling, repulsion_penalty, dynamics_cost_scaling):
        self.n_agents = n_agents
        self.grid_side = grid_side
        self.dynamics = dynamics
        self.per_uav_actions = per_uav_actions
        self.discount = discount
        self.goal_regions = goal_regions
        self.region_to_uavids = region_to_uavids
        self.uav_to_region_map = uav_to_region_map
        self.constant_cg_adj_mat = constant_cg_adj_mat
        self.reach_goal_bonus = reach_goal_bonus
        self.proximity_penalty_scaling = proximity_penalty_scaling
        self.repulsion_penalty = repulsion_penalty
        self.dynamics_cost_scaling = dynamics_cost_scaling


class MultiUAVDelivery:
    def __init__(self, n_agents, pset, rewset):
        self.n_agents = n_agents
        self.n_actions_per_agent = 10
        self.pset = pset[self.n_agents]
        self.rewset = rewset
        self.mdp = self.FirstOrderMultiUAVDelivery(self.n_agents, self.pset, self.rewset)
        self.s0 = None
        self.min_action = 0
        self.max_action = 9

    def FirstOrderMultiUAVDelivery(self, n_agents, pset, rewset, seed=7):
        # It initializes the parameters of the environment
        uavparams = UAVParameters(XY_AXIS_RES=pset['XY_AXIS_RES'],
                                  XYDOT_LIM=pset['XYDOT_LIM'],
                                  XYDOT_STEP=pset['XYDOT_STEP'],
                                  PROXIMITY_THRESH=1.5 * pset['XY_AXIS_RES'],
                                  CG_PROXIMITY_THRESH=3.0 * pset['XY_AXIS_RES'])

        # It defines the dynamics of the environment
        dynamics = FirstOrderUAVDynamics(timestep=1.0,
                                         noise=(np.array([pset['NOISESTD'], pset['NOISESTD']]),),
                                         params=uavparams)

        # It yields the goal regions and assigns the UAVs to the goal regions
        goal_regions, region_to_uavids = get_quadrant_goal_regions(n_agents, pset['XY_AXIS_RES'])

        mdp = self.MultiUAVDeliveryMDP(n_agents=n_agents, dynamics=dynamics,
                                       goal_regions=goal_regions, region_to_uavids=region_to_uavids,
                                       goal_bonus=rewset[0], prox_scaling=rewset[1],
                                       repul_pen=rewset[2], dyn_scaling=rewset[3])

        return mdp

    # Constructor for MDP that takes minimal required info and does the rest in place
    def MultiUAVDeliveryMDP(self, n_agents, dynamics, discount=1.0, goal_regions=None,
                            region_to_uavids=None, goal_bonus=None, prox_scaling=None,
                            repul_pen=None, dyn_scaling=None):
        per_uav_actions = self.get_per_uav_actions(dynamics)
        grid_side = int((2 * dynamics.params.XY_LIM) / dynamics.params.XY_AXIS_RES)
        uav_to_region_map = [0] * n_agents
        for reg_idx, uavset in enumerate(region_to_uavids):
            for uavid in uavset:
                uav_to_region_map[uavid] = reg_idx

        # It defines the adj. matrix
        cg_adj_mat = np.zeros((n_agents, n_agents), dtype=int)
        for reg_uavset in region_to_uavids:
            for i in reg_uavset:
                for j in reg_uavset:
                    if i != j:
                        cg_adj_mat[i, j] = 1
                        cg_adj_mat[j, i] = 1

        return MultiUAVDeliveryMDPConstructor(n_agents, grid_side, dynamics, per_uav_actions, discount,
                                              goal_regions, region_to_uavids, uav_to_region_map, cg_adj_mat,
                                              goal_bonus, prox_scaling, repul_pen, dyn_scaling)

    def coordination_graph(self, p, s):
        return nx.Graph(self.coord_graph_adj_mat(p, s))

    def coord_graph_adj_mat(self, p, s):
        state_cg_mat = p.constant_cg_adj_mat.copy()
        for i, si in enumerate(s):
            for j, sj in enumerate(s):
                if i != j and p.uav_to_region_map[i] != p.uav_to_region_map[j] and \
                        self.gc_norm(p.dynamics.params, si.coords, sj.coords) <= p.dynamics.params.CG_PROXIMITY_THRESH:
                    state_cg_mat[i, j] = 1
                    state_cg_mat[j, i] = 1
        return state_cg_mat

    def initialstate(self, p):
        initstate = []
        i = 1
        while i <= p.n_agents:
            rand_gc = self.xy_to_grid_coords(p.dynamics.params, generate_start_state(p.dynamics))
            invalid_coords = True
            for reg in p.goal_regions:
                if self.is_in_region(p.dynamics.params, reg, rand_gc) or any(
                        UAVGeneralState(rand_gc, False) == s for s in initstate):
                    invalid_coords = False
                    break
            if invalid_coords:
                initstate.append(UAVGeneralState(rand_gc, False))
                i += 1
        return initstate

    def print_network(self, coordgraph):
        nx.draw(coordgraph, with_labels=True)
        plt.show()

    def step(self, p, s, a_int):
        a = self.int_to_action(a_int)
        coordgraph = self.coordination_graph(p, s)
        # self.print_network(coordgraph)
        sp_vec = [UAVGeneralState(0, False) for _ in range(self.n_agents)]

        # NOTE: Local reward vector, different for each agent
        r_vec = [None] * self.n_agents

        # Simple way of doing this:
        # Consider every agent at a time (so focusing on its reward and next state)
        # Consider each of its neighbors in the CG
        # If it tries to board, check that no neighbor is in threshold and if successful, get boarding bonus (only that Agent?), else boarding penalty?
        # If it tries to move, check if any neighbor prevents it (repulsion penalty), otherwise dynamics cost
        # If it has boarded, it should only be no-op and not change it's indiv state (0,0)
        terminal = False
        n_prox = 0
        n_coll = 0
        for idx in range(self.n_agents):
            s_idx = s[idx]
            a_idx = a[idx]
            nbrs = list(coordgraph.neighbors(idx))

            if a_idx.no_op:
                # assert s_idx.boarded == True, f"No-op action taken by agent {idx} in state " \
                #                               f"{(s_idx.coords, s_idx.boarded)}"
                sp_vec[idx] = s_idx
                r_vec[idx] = 0.0
            elif a_idx.to_board:
                # assert (s_idx.boarded == False and is_in_region(p.dynamics.params,
                #                                                 p.goal_regions[p.uav_to_region_map[idx]],
                #                                                 s_idx.coords)), \
                #     f"Board action taken by agent {idx} of state {s_idx.coords} " \
                #     f"when not in region {p.goal_regions[p.uav_to_region_map[idx]].cen}"
                any_nbr_boarding = False
                # If neighbor in same goal is boarding, prevent and penalize
                for n in nbrs:
                    if a[n].to_board and p.uav_to_region_map[n] == p.uav_to_region_map[idx]:
                        any_nbr_boarding = True
                        break
                # Repulsion penalty; stay in-place
                if any_nbr_boarding:
                    sp_vec[idx] = s_idx
                    r_vec[idx] = -p.repulsion_penalty
                    n_coll += 1
                else:
                    # Successful boarding; same location and board true
                    sp_vec[idx] = UAVGeneralState(s_idx.coords, True)
                    r_vec[idx] = p.reach_goal_bonus
                    # terminal = True
            # Dynamics action
            else:
                t_coords = self.grid_coords_to_xy(p.dynamics.params, s_idx.coords)
                temp_new_uav_state = next_uav_state(p.dynamics,
                                                    FirstOrderUAVState(t_coords[0], t_coords[1]),
                                                    a_idx.dyn_action)
                new_coords = self.xy_to_grid_coords(p.dynamics.params, temp_new_uav_state)
                sp_vec[idx] = UAVGeneralState(new_coords, False)
                r_vec[idx] = -p.dynamics_cost_scaling * dynamics_cost(p.dynamics, a_idx.dyn_action)

        # Do another loop over cells and if any two in same cell, move all of them back to their original cell
        any_clashing_positions = True
        while any_clashing_positions:
            clashing_positions = {}
            for i, si in enumerate(sp_vec):
                # si_coord_idx = np.ravel_multi_index(si.coords, (p.grid_side, p.grid_side))
                if si.coords not in clashing_positions:
                    clashing_positions[si.coords] = {i}
                else:
                    clashing_positions[si.coords].add(i)

            any_clashing_positions = False
            for coord_idx, agents in clashing_positions.items():
                if len(agents) > 1:
                    any_clashing_positions = True
                    for agt in agents:
                        sp_vec[agt] = s[agt]
                        r_vec[agt] = -p.repulsion_penalty
                        n_coll += 1

        # Now we are guaranteed no clashing, loop over states and add reward if closer to goal
        for idx, (si, spi) in enumerate(zip(s, sp_vec)):
            idx_goal = p.goal_regions[p.uav_to_region_map[idx]]
            rel_dist = np.linalg.norm(
                np.array(self.grid_coords_to_xy(p.dynamics.params, si.coords)) - np.array(idx_goal.cen)) - \
                       np.linalg.norm(
                           np.array(self.grid_coords_to_xy(p.dynamics.params, spi.coords)) - np.array(idx_goal.cen))
            if abs(rel_dist) > p.dynamics.params.XY_AXIS_RES / 2:
                r_vec[idx] += p.proximity_penalty_scaling * (1.0 / rel_dist)
                n_prox += 1

        # Finally, loop through sp and add a penalty for any pair too close to each other
        for i in range(self.n_agents - 1):
            if not sp_vec[i].boarded:
                for j in range(i + 1, self.n_agents):
                    if not sp_vec[j].boarded:
                        dist = self.gc_norm(p.dynamics.params, sp_vec[j].coords, sp_vec[i].coords)
                        assert dist > p.dynamics.params.XY_AXIS_RES / 2.0, \
                            f"Dist between ({i}, {j}) of ({sp_vec[i].coords}, {sp_vec[i].boarded}) " \
                            f"and ({sp_vec[j].coords}, {sp_vec[j].boarded}) is 0!"
                        if dist <= p.dynamics.params.PROXIMITY_THRESH:
                            r_vec[i] -= p.proximity_penalty_scaling * (1.0 / dist)
                            r_vec[j] -= p.proximity_penalty_scaling * (1.0 / dist)
                            n_prox += 2

        # return sp_vec, r_vec, terminal, {"proximity": n_prox, "collisions": n_coll}
        # print_grid()
        # for i in range(8):
        #     print('Agent %s' % i)
        #     print(s[i])
        #     print(a_int[i])
        #     print(a[i])
        #     print(sp_vec[i])
        #     print('\n')
        return sp_vec, np.array(r_vec), terminal

    def reward_function(self, s, a_int):
        a = self.int_to_action(a_int)
        coordgraph = self.coordination_graph(self.mdp, s[:-1])
        # self.print_network(coordgraph)
        sp_vec = [UAVGeneralState(0, True) if s[aidx].boarded == True else UAVGeneralState(0, False)
                  for aidx in range(self.n_agents)]

        # NOTE: Local reward vector, different for each agent
        r_vec = [0.0] * self.n_agents

        # Simple way of doing this:
        # Consider every agent at a time (so focusing on its reward and next state)
        # Consider each of its neighbors in the CG
        # If it tries to board, check that no neighbor is in threshold and if successful, get boarding bonus (only that Agent?), else boarding penalty?
        # If it tries to move, check if any neighbor prevents it (repulsion penalty), otherwise dynamics cost
        # If it has boarded, it should only be no-op and not change it's indiv state (0,0)
        terminal = False

        for idx in range(self.n_agents):
            s_idx = s[idx]
            a_idx = a[idx]
            nbrs = list(coordgraph.neighbors(idx))

            if a_idx.no_op:
                # assert s_idx.boarded == True, f"No-op action taken by agent {idx} in state " \
                #                               f"{(s_idx.coords, s_idx.boarded)}"
                sp_vec[idx] = s_idx
                r_vec[idx] = 0.0
            elif a_idx.to_board:
                # assert (s_idx.boarded == False and is_in_region(p.dynamics.params,
                #                                                 p.goal_regions[p.uav_to_region_map[idx]],
                #                                                 s_idx.coords)), \
                #     f"Board action taken by agent {idx} of state {s_idx.coords} " \
                #     f"when not in region {p.goal_regions[p.uav_to_region_map[idx]].cen}"
                any_nbr_boarding = False
                # If neighbor in same goal is boarding, prevent and penalize
                for n in nbrs:
                    if a[n].to_board and self.mdp.uav_to_region_map[n] == self.mdp.uav_to_region_map[idx]:
                        # print('Repulsion penalty; stay in-place')
                        any_nbr_boarding = True
                        break
                # Repulsion penalty; stay in-place
                if any_nbr_boarding:
                    sp_vec[idx] = s_idx
                    r_vec[idx] = -self.mdp.repulsion_penalty
                else:
                    # Successful boarding; same location and board true
                    sp_vec[idx] = UAVGeneralState(s_idx.coords, True)
                    r_vec[idx] = self.mdp.reach_goal_bonus
                    # terminal = True
            # Dynamics action
            else:
                t_coords = self.grid_coords_to_xy(self.mdp.dynamics.params, s_idx.coords)
                temp_new_uav_state = next_uav_state(self.mdp.dynamics,
                                                    FirstOrderUAVState(t_coords[0], t_coords[1]),
                                                    a_idx.dyn_action)
                new_coords = self.xy_to_grid_coords(self.mdp.dynamics.params, temp_new_uav_state)
                sp_vec[idx] = UAVGeneralState(new_coords, False)
                r_vec[idx] = -self.mdp.dynamics_cost_scaling * dynamics_cost(self.mdp.dynamics, a_idx.dyn_action)


        # Do another loop over cells and if any two in same cell, move all of them back to their original cell
        any_clashing_positions = True
        while any_clashing_positions:
            clashing_positions = {}
            for i, si in enumerate(sp_vec):
                # si_coord_idx = np.ravel_multi_index(si.coords, (p.grid_side, p.grid_side))
                # if si.boarded:
                #     continue
                if si.coords not in clashing_positions:
                    clashing_positions[si.coords] = {i}
                else:
                    clashing_positions[si.coords].add(i)

            any_clashing_positions = False
            for coord_idx, agents in clashing_positions.items():
                if len(agents) > 1:
                    any_clashing_positions = True
                    for agt in agents:
                        sp_vec[agt] = s[agt]
                        r_vec[agt] = -self.mdp.repulsion_penalty

            any_clashing_positions = False
        # Now we are guaranteed no clashing, loop over states and add reward if closer to goal
        for idx, (si, spi) in enumerate(zip(s, sp_vec)):
            idx_goal = self.mdp.goal_regions[self.mdp.uav_to_region_map[idx]]
            rel_dist = np.linalg.norm(
                np.array(self.grid_coords_to_xy(self.mdp.dynamics.params, si.coords)) - np.array(idx_goal.cen)) - \
                       np.linalg.norm(
                           np.array(self.grid_coords_to_xy(self.mdp.dynamics.params, spi.coords)) - np.array(idx_goal.cen))
            if abs(rel_dist) > self.mdp.dynamics.params.XY_AXIS_RES / 2:
                r_vec[idx] += self.mdp.proximity_penalty_scaling * (1.0 / rel_dist)

        # Finally, loop through sp and add a penalty for any pair too close to each other
        for i in range(self.n_agents - 1):
            if not sp_vec[i].boarded:
                for j in range(i + 1, self.n_agents):
                    if not sp_vec[j].boarded:
                        dist = self.gc_norm(self.mdp.dynamics.params, sp_vec[j].coords, sp_vec[i].coords)
                        if dist <= self.mdp.dynamics.params.PROXIMITY_THRESH:
                            if dist <= 0.001:
                                r_vec[i] -= self.mdp.proximity_penalty_scaling * 20
                                r_vec[j] -= self.mdp.proximity_penalty_scaling * 20
                            else:
                                r_vec[i] -= self.mdp.proximity_penalty_scaling * (1.0 / dist)
                                r_vec[j] -= self.mdp.proximity_penalty_scaling * (1.0 / dist)

        return np.array(r_vec)

    def reset(self, ):
        self.s0 = self.initialstate(self.mdp)

        return self.s0

    def int_to_action(self, joint_action_int):

        joint_action = []
        for a_i in joint_action_int:
            joint_action.append(int_to_act[a_i])

        return joint_action

    def action_to_int(self, action):

        return act_to_int[str(action)]

    ## Discretize continuous x-y state to lattice coords (x, y)
    def xy_to_grid_coords(self, params, cont_state):
        xcoord = int(np.ceil((cont_state.x + params.XY_LIM) / params.XY_AXIS_RES))
        ycoord = int(np.ceil((cont_state.y + params.XY_LIM) / params.XY_AXIS_RES))

        if xcoord == 0:
            xcoord = 1
        if ycoord == 0:
            ycoord = 1

        assert xcoord * ycoord != 0, f"State {cont_state} maps to coords ({xcoord},{ycoord})"

        return xcoord, ycoord

    def gc_norm(self, params, gc1, gc2):
        xy1 = self.grid_coords_to_xy(params, gc1)
        xy2 = self.grid_coords_to_xy(params, gc2)
        return np.linalg.norm(np.array(xy1) - np.array(xy2))

    def is_in_region(self, params, reg, gc):
        cen = np.array(reg.cen)
        xy_coords = self.grid_coords_to_xy(params, gc)
        return np.linalg.norm(cen - xy_coords) <= reg.rad

    ## NOTE: (x,y) returned as a Float64 Tuple. It must then be used according to UAVState type
    def grid_coords_to_xy(self, params, coords):
        x = -params.XY_LIM + params.XY_AXIS_RES * ((2 * coords[0] - 1) / 2.0)
        y = -params.XY_LIM + params.XY_AXIS_RES * ((2 * coords[1] - 1) / 2.0)
        return x, y

    ## To define full set of actions per UAV
    def get_per_uav_actions(self, dynamics):
        uav_ctl_actions = get_uav_control_actions(dynamics)
        atype = type(uav_ctl_actions[0])

        per_uav_actions = []

        del uav_ctl_actions[4]
        # First fill up with control actions
        for uca in uav_ctl_actions:
            # if uca.xdot != 0.0 and uca.ydot != 0.0:
            per_uav_actions.append(UAVGeneralAction(uca, False, False))

        # Now push BOARD
        per_uav_actions.append(UAVGeneralAction(atype(), True, False))

        # Then push NO-OP
        per_uav_actions.append(UAVGeneralAction(atype(), False, True))

        return per_uav_actions

    # Actions are [dynamics ... BOARD NO-OP]
    def agent_actions(self, p, idx, s):
        # If boarded, only no-op action can be taken
        if s[idx].boarded:
            return [p.per_uav_actions[-1]]

        # Otherwise, loop over regions and check if drone is in it
        for reg_idx, reg in enumerate(p.goal_regions):
            if self.is_in_region(p.dynamics.params, reg, s[idx].coords) and p.uav_to_region_map[idx] == reg_idx:
                # Then can do dynamics as well as board; only exclude NO-OP
                return p.per_uav_actions[:-1]

        # Otherwise, only dynamics; exclude BOARD as well as NO-OP
        return p.per_uav_actions[:-2]

    # Actions are [dynamics ... BOARD NO-OP]
    def single_agent_actions(self, p, idx, s):
        # If boarded, only no-op action can be taken
        if s.boarded:
            return [p.per_uav_actions[-1]]

        # Otherwise, loop over regions and check if drone is in it
        for reg_idx, reg in enumerate(p.goal_regions):
            if self.is_in_region(p.dynamics.params, reg, s.coords) and p.uav_to_region_map[idx] == reg_idx:
                # Then can do dynamics as well as board; only exclude NO-OP
                return p.per_uav_actions[:-1]

        # Otherwise, only dynamics; exclude BOARD as well as NO-OP
        return p.per_uav_actions[:-2]

    def str_to_state(self, state):
        return state.lstrip("\'").rstrip("\'")


def agent_actionindex(p, idx, a):
    return p.per_uav_actions.index(a)


def actions(p):
    return list(product(*([p.per_uav_actions] * n_agents(p))))


def actionindex(p, a):
    return p.per_uav_actions.index(a)


def agent_states(p, idx):
    coordsset = [GridCoords(x, y) for y in range(1, p.grid_side + 1) for x in range(1, p.grid_side + 1)]
    stateset = []
    for coords in coordsset:
        stateset.append(UAVGeneralState(coords, False))
        stateset.append(UAVGeneralState(coords, True))
    return stateset


def agent_stateindex(p, idx, s):
    coord_idx = np.ravel_multi_index(s.coords, (p.grid_side, p.grid_side))
    if s.boarded:
        return 2 * coord_idx
    else:
        return 2 * coord_idx - 1


def is_terminal(s):
    return all(agent.boarded for agent in s)


if __name__ == '__main__':
    act_to_int = {
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=-0.2, ydot=-0.2), to_board=False, no_op=False)': 0,
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=-0.2, ydot=0.0), to_board=False, no_op=False)': 1,
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=-0.2, ydot=0.2), to_board=False, no_op=False)': 2,
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.0, ydot=-0.2), to_board=False, no_op=False)': 3,
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.0, ydot=0.2), to_board=False, no_op=False)': 4,
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.2, ydot=-0.2), to_board=False, no_op=False)': 5,
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.2, ydot=0.0), to_board=False, no_op=False)': 6,
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.2, ydot=0.2), to_board=False, no_op=False)': 7,
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.0, ydot=0.0), to_board=True, no_op=False)': 8,
        'UAVGeneralAction(dyn_action=FirstOrderUAVAction(xdot=0.0, ydot=0.0), to_board=False, no_op=True)': 9
    }


    def agent_actions(env, p, idx, s):
        # If boarded, only no-op action can be taken
        if s.boarded:
            return [p.per_uav_actions[-1]]

        # Otherwise, loop over regions and check if drone is in it
        for reg_idx, reg in enumerate(p.goal_regions):
            if env.is_in_region(p.dynamics.params, reg, s.coords) and p.uav_to_region_map[idx] == reg_idx:
                # Then can do dynamics as well as board; only exclude NO-OP
                return p.per_uav_actions[:-1]

        # Otherwise, only dynamics; exclude BOARD as well as NO-OP
        return p.per_uav_actions[:-2]
    n_agents = 8
    PSET = {
        8: {"XY_AXIS_RES": 0.2, "XYDOT_LIM": 0.2, "XYDOT_STEP": 0.2, "NOISESTD": 0.1},
        16: {"XY_AXIS_RES": 0.1, "XYDOT_LIM": 0.1, "XYDOT_STEP": 0.1, "NOISESTD": 0.05},
        32: {"XY_AXIS_RES": 0.08, "XYDOT_LIM": 0.08, "XYDOT_STEP": 0.08, "NOISESTD": 0.05},
        48: {"XY_AXIS_RES": 0.05, "XYDOT_LIM": 0.05, "XYDOT_STEP": 0.05, "NOISESTD": 0.02}
    }
    # Reward = goal_bonus, prox_scaling, repul_pen, dynamics_scaling
    rewset = (1000.0, 1.0, 10.0, 10.0)
    env = MultiUAVDelivery(n_agents, PSET, rewset)
    state = env.reset()
    for _ in range(10):
        joint_action = [random.choice(agent_actions(env, env.mdp, agent, state[agent])) for agent in
                        range(env.mdp.n_agents)]
        joint_action_to_int = [act_to_int[str(a)] for a in joint_action]
        _, reward, info = env.step(env.mdp, state, joint_action_to_int)

        reward_2 = env.reward_function(state, joint_action_to_int)
