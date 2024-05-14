import random

import numpy as np

from envs.multiUAVDelivery import MultiUAVDelivery


class MultiAgentSysAdminGenerativeBaselinePolicy:
    def __init__(self, env, gamma, method='generative', epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.nb_states = env.nb_states
        self.nb_actions = env.nb_actions
        self.epsilon = epsilon
        self.method = method
        self.p_reboot = 0.6
        self.p_no_op = 0.6

    # Get joint probabilities for joint-action
    def get_prob_of_baseline(self, state):
        prob = np.zeros((self.env.n_agents, 2))
        for aidx in range(self.env.n_agents):
            if state[aidx][0] == 3:
                prob[aidx, 1] = self.p_reboot
                prob[aidx, 0] = 1.0 - prob[aidx, 1]
            else:
                prob[aidx, 0] = self.p_no_op
                prob[aidx, 1] = 1.0 - prob[aidx, 0]

        return prob

    # Get joint-action
    def get_act_from_baseline(self, state):
        p_baseline = self.get_prob_of_baseline(state)

        joint_act = []
        for ai in range(self.env.number_of_agents):
            joint_act.append(np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline[ai]))

        return joint_act

    # Get probabilities for a single action
    def get_single_prob_of_baseline(self, state, agent):
        prob = np.zeros(self.env.n_actions_per_agent)
        if state[0] == 3:
            prob[1] = self.p_reboot
            prob[0] = 1.0 - prob[1]
        else:
            prob[0] = self.p_no_op
            prob[1] = 1.0 - prob[0]

        return prob

    # Get single action
    def get_single_act_from_baseline(self, state, agent):
        p_baseline = self.get_single_prob_of_baseline(state, 0)

        return np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline)


class MultiUAVDeliveryGenerativeBaselinePolicy:
    def __init__(self, n_agents, env, method='generative'):
        self.n_agents = n_agents
        self.method = method
        self.p_err_boarding = 0.4
        self.p_err_moving = 0.4
        self.env = env
        self.goal_regions = [reg for reg in self.env.mdp.goal_regions]
        self.uav_to_region_map = self.env.mdp.uav_to_region_map
        self.params = self.env.mdp.dynamics.params
        self.reg_to_int = self.reg_to_int()
        self.coords_to_regs = self.map_coords_to_regions()

    def get_prob_of_baseline(self, state):
        prob = np.zeros((self.n_agents, self.env.n_actions_per_agent))
        for aidx in range(self.n_agents):
            if not state[aidx].coords in self.coords_to_regs.keys():
                prob[aidx, :8] = 1 / 8
            else:
                if not state[aidx].boarded and self.env.is_in_region(self.params, self.goal_regions[
                    self.uav_to_region_map[aidx]], state[aidx].coords):
                    prob[aidx][8] = 1 - self.p_err_boarding
                    prob[aidx][:8] = self.p_err_boarding / 8
                elif state[aidx].boarded:
                    prob[aidx][9] = 1.0
                else:
                    # Goal region for UAV i is 0
                    if self.coords_to_regs[state[aidx].coords] == 0:
                        # UAV i is in region 0 (inside the goal region)
                        if self.uav_to_region_map[aidx] == 0:
                            prob[aidx][9] = 1.0
                        # UAV i is in region 1 move SOUTH
                        elif self.uav_to_region_map[aidx] == 1:
                            prob[aidx][6] = 1 - self.p_err_moving
                            indexes = [0, 1, 2, 3, 4, 5, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                        # UAV i is in region 2 move SOUTH-EAST
                        elif self.uav_to_region_map[aidx] == 2:
                            prob[aidx][7] = 1 - self.p_err_moving
                            indexes = [0, 1, 2, 3, 4, 5, 6]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                        # UAV i is in region 3 move EAST
                        else:
                            prob[aidx][4] = 1 - self.p_err_moving
                            indexes = [0, 1, 2, 3, 5, 6, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                    # Goal region for UAV i is 1
                    elif self.coords_to_regs[state[aidx].coords] == 1:
                        # UAV i is in region 0 move NORTH
                        if self.uav_to_region_map[aidx] == 0:
                            prob[aidx][1] = 1 - self.p_err_moving
                            indexes = [0, 2, 3, 4, 5, 6, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                        # UAV i is in region 1 (inside the goal region)
                        elif self.uav_to_region_map[aidx] == 1:
                            prob[aidx][9] = 1.0
                        # UAV i is in region 2 move EAST
                        elif self.uav_to_region_map[aidx] == 2:
                            prob[aidx][4] = 1 - self.p_err_moving
                            indexes = [0, 1, 2, 3, 5, 6, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                        # UAV i is in region 3 move NORTH-EAST
                        else:
                            prob[aidx][2] = 1 - self.p_err_moving
                            indexes = [0, 1, 3, 4, 5, 6, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                    # Goal region for UAV i is 2
                    elif self.coords_to_regs[state[aidx].coords] == 2:
                        # UAV i is in region 0 move NORTH-WEST
                        if self.uav_to_region_map[aidx] == 0:
                            prob[aidx][0] = 1 - self.p_err_moving
                            indexes = [1, 2, 3, 4, 5, 6, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                        # UAV i is in region 1 move WEST
                        elif self.uav_to_region_map[aidx] == 1:
                            prob[aidx][3] = 1 - self.p_err_moving
                            indexes = [0, 1, 2, 4, 5, 6, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                        # UAV i is in region 2 (inside the goal region)
                        elif self.uav_to_region_map[aidx] == 2:
                            prob[aidx][9] = 1.0
                        # UAV i is in region 3 move NORTH
                        else:
                            prob[aidx][1] = 1 - self.p_err_moving
                            indexes = [0, 2, 3, 4, 5, 6, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                    # Goal region for UAV i is 3
                    else:
                        # UAV i is in region 0 move WEST
                        if self.uav_to_region_map[aidx] == 0:
                            prob[aidx][3] = 1 - self.p_err_moving
                            indexes = [0, 1, 2, 4, 5, 6, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                        # UAV i is in region 1 move SOUTH-WEST
                        elif self.uav_to_region_map[aidx] == 1:
                            prob[aidx][5] = 1 - self.p_err_moving
                            indexes = [0, 1, 2, 3, 4, 6, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                        # UAV i is in region 2 move SOUTH
                        elif self.uav_to_region_map[aidx] == 2:
                            prob[aidx][6] = 1 - self.p_err_moving
                            indexes = [0, 1, 2, 3, 4, 5, 7]
                            prob[aidx][indexes] = self.p_err_moving / 7
                            prob[aidx] /= sum(prob[aidx])
                        # UAV i is in region 3 (inside the goal region)
                        else:
                            prob[aidx][9] = 1.0
        return prob

    def get_act_from_baseline(self, state):
        p_baseline = self.get_prob_of_baseline(state)

        joint_act = []
        for ai in range(self.env.n_agents):
            joint_act.append(np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline[ai]))

        return joint_act

    # Get probabilities for a single action
    def get_single_prob_of_baseline(self, state, aidx):
        state = state[0]
        prob = np.zeros(self.env.n_actions_per_agent)
        if not state.coords in self.coords_to_regs.keys():
            prob[:8] = 1 / 8
        else:
            if not state.boarded and self.env.is_in_region(self.params, self.goal_regions[
                self.uav_to_region_map[aidx]], state.coords):
                prob[8] = 1 - self.p_err_boarding
                prob[:8] = self.p_err_boarding / 8
            elif state.boarded:
                prob[9] = 1.0
            else:
                # Goal region for UAV i is 0
                if self.coords_to_regs[state.coords] == 0:
                    # UAV i is in region 0 (inside the goal region)
                    if self.uav_to_region_map[aidx] == 0:
                        prob[9] = 1.0
                    # UAV i is in region 1 move SOUTH
                    elif self.uav_to_region_map[aidx] == 1:
                        prob[6] = 1 - self.p_err_moving
                        indexes = [0, 1, 2, 3, 4, 5, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                    # UAV i is in region 2 move SOUTH-EAST
                    elif self.uav_to_region_map[aidx] == 2:
                        prob[7] = 1 - self.p_err_moving
                        indexes = [0, 1, 2, 3, 4, 5, 6]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                    # UAV i is in region 3 move EAST
                    else:
                        prob[4] = 1 - self.p_err_moving
                        indexes = [0, 1, 2, 3, 5, 6, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                # Goal region for UAV i is 1
                elif self.coords_to_regs[state.coords] == 1:
                    # UAV i is in region 0 move NORTH
                    if self.uav_to_region_map[aidx] == 0:
                        prob[1] = 1 - self.p_err_moving
                        indexes = [0, 2, 3, 4, 5, 6, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                    # UAV i is in region 1 (inside the goal region)
                    elif self.uav_to_region_map[aidx] == 1:
                        prob[9] = 1.0
                    # UAV i is in region 2 move EAST
                    elif self.uav_to_region_map[aidx] == 2:
                        prob[4] = 1 - self.p_err_moving
                        indexes = [0, 1, 2, 3, 5, 6, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                    # UAV i is in region 3 move NORTH-EAST
                    else:
                        prob[2] = 1 - self.p_err_moving
                        indexes = [0, 1, 3, 4, 5, 6, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                # Goal region for UAV i is 2
                elif self.coords_to_regs[state.coords] == 2:
                    # UAV i is in region 0 move NORTH-WEST
                    if self.uav_to_region_map[aidx] == 0:
                        prob[0] = 1 - self.p_err_moving
                        indexes = [1, 2, 3, 4, 5, 6, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                    # UAV i is in region 1 move WEST
                    elif self.uav_to_region_map[aidx] == 1:
                        prob[3] = 1 - self.p_err_moving
                        indexes = [0, 1, 2, 4, 5, 6, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                    # UAV i is in region 2 (inside the goal region)
                    elif self.uav_to_region_map[aidx] == 2:
                        prob[9] = 1.0
                    # UAV i is in region 3 move NORTH
                    else:
                        prob[1] = 1 - self.p_err_moving
                        indexes = [0, 2, 3, 4, 5, 6, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                # Goal region for UAV i is 3
                else:
                    # UAV i is in region 0 move WEST
                    if self.uav_to_region_map[aidx] == 0:
                        prob[3] = 1 - self.p_err_moving
                        indexes = [0, 1, 2, 4, 5, 6, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                    # UAV i is in region 1 move SOUTH-WEST
                    elif self.uav_to_region_map[aidx] == 1:
                        prob[5] = 1 - self.p_err_moving
                        indexes = [0, 1, 2, 3, 4, 6, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                    # UAV i is in region 2 move SOUTH
                    elif self.uav_to_region_map[aidx] == 2:
                        prob[6] = 1 - self.p_err_moving
                        indexes = [0, 1, 2, 3, 4, 5, 7]
                        prob[indexes] = self.p_err_moving / 7
                        prob /= sum(prob)
                    # UAV i is in region 3 (inside the goal region)
                    else:
                        prob[9] = 1.0
        return prob

    # Get single action
    def get_single_act_from_baseline(self, state, agent):
        p_baseline = self.get_single_prob_of_baseline(state, agent)

        return np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline)

    def reg_to_int(self):
        reg_to_int = dict()
        for i, reg in enumerate(self.goal_regions):
            reg_to_int[str(reg.cen)] = i

        return reg_to_int

    def map_coords_to_regions(self):
        coords_to_regs = dict()
        for x in range(0, 11):
            for y in range(0, 11):
                for reg in self.goal_regions:
                    if self.env.is_in_region(self.params, reg, (x, y)):
                        coords_to_regs[(x, y)] = self.reg_to_int[str(reg.cen)]

        return coords_to_regs


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
    pi_b = MultiUAVDeliveryGenerativeBaselinePolicy(n_agents, env)
    state = env.reset()
    for _ in range(10):
        joint_action = [random.choice(agent_actions(env, env.mdp, agent, state[agent])) for agent in
                        range(env.mdp.n_agents)]
        joint_action_to_int = [act_to_int[str(a)] for a in joint_action]
        state, reward, info = env.step(env.mdp, state, joint_action_to_int)
