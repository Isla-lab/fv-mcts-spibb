import itertools

import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt

"""
- status: good = 1, fail = 2, dead = 3
- load: idle = 1, work = 2, done = 3
- `p_fail_base`:
- `p_fail_bonus`:
- `p_dead_base`:
- `p_dead_bonus`:
- `p_load`:  probability of getting a job when idle.
- `p_doneG`: probability of completing a job when good.
- `p_doneF`: probability of completing a job when faulty.
`p_fail_bonus` and `p_dead_bonus` are additional bonuses counted
when all neighbors are faulty. Counted per agent.
If a machine with 2 neighbors has a single faulty neighbor, it will get
an additional failing probability of `p_fail_bonus/2`. If the same machine
has one faulty neighbor and one dead neighbor, it will get a penalty of
`p_fail_bonus/2 + p_dead_bonus/2`.
"""


class MultiAgentSysAdmin:
    def __init__(self, n_agents, type_of_network='UniSysAdmin'):
        self.type_of_network = type_of_network
        self.n_agents = n_agents
        # status
        self.p_fail_base = 0.4
        self.p_fail_bonus = 0.2
        self.p_dead_base = 0.1
        self.p_dead_bonus = 0.5
        # load
        self.p_load = 0.6
        self.p_doneG = 0.9
        self.p_doneF = 0.6
        self.discount = 0.9
        self.reboot_penalty = -0.0
        if type_of_network == 'UniSysAdmin':
            self.reboot_penalty = -0.0
        if type_of_network == 'RingofRingSysAdmin':
            self.n_rings = 3
            self.n_agents_per_ring = 4
            self.n_agents = self.n_agents_per_ring * self.n_rings
        if type_of_network == 'RandomSysAdmin':
            self.n_edges = 5
            self.seed = 123
        self.nb_states = pow(3, self.n_agents)
        self.nb_actions = pow(2, self.n_agents)
        self.n_actions_per_agent = 2
        self.min_action = 0
        self.max_action = 1
        self.state = None
        self.coordgraph = self.coordination_graph()
        self.mdp = None

    def coord_graph_adj_mat(self, ):
        if self.type_of_network == 'UniSysAdmin':
            mat = np.zeros((self.n_agents, self.n_agents), dtype=int)
            for i in range(self.n_agents - 1):
                mat[i, i + 1] = 1
            mat[self.n_agents - 1, 0] = 1

            return mat

        elif self.type_of_network == 'BiSysAdmin':
            mat = np.zeros((self.n_agents, self.n_agents), dtype=int)
            for i in range(self.n_agents - 1):
                mat[i, i + 1] = 1
            mat[self.n_agents - 1, 0] = 1

            mat[self.n_agents - 1, 1] = 1
            mat = mat + mat.T

            return mat

        elif self.type_of_network == 'RingofRingSysAdmin':
            na = self.n_agents
            mat = np.zeros((na, na), dtype=int)
            # Inner ring
            for idx in itertools.product(range(1, na + 1, self.n_rings), range(1, na + 1, self.n_rings)):
                if idx[0] == idx[1]:
                    continue
                mat[idx] = 1

            for i in range(0, na - 1, self.n_rings):
                for j in range(i, i + self.n_agents_per_ring - 1):
                    mat[j - 1, j] = 1
                    mat[j, j - 1] = 1
                mat[i, i + self.n_agents_per_ring - 2] = 1
                mat[i + self.n_agents_per_ring - 2, i] = 1

            return mat

        elif self.type_of_network == 'RandomSysAdmin':
            return nx.fast_gnp_random_graph(self.n_agents, self.n_edges, seed=self.seed, directed=False)
        elif self.type_of_network == 'StarSysAdmin':
            mat = np.zeros((self.n_agents, self.n_agents), dtype=int)
            for i in range(1, self.n_agents):
                mat[0, i] = 1
                mat[i, 0] = 1

            return mat

    def coordination_graph(self, ):
        if self.type_of_network == 'UniSysAdmin':
            return nx.Graph(self.coord_graph_adj_mat())
        elif self.type_of_network == 'BiSysAdmin':
            return nx.Graph(self.coord_graph_adj_mat())
        elif self.type_of_network == 'RandomSysAdmin':
            return self.coord_graph_adj_mat()
        elif self.type_of_network == 'RingofRingSysAdmin':
            return nx.Graph(self.coord_graph_adj_mat())
        elif self.type_of_network == 'StarSysAdmin':
            return nx.Graph(self.coord_graph_adj_mat())

    def reward_function(self, state, action):
        total_reward = [None] * self.n_agents
        for aidx in range(self.n_agents):
            reward = 0.0
            bonus = 0.0
            neighs = list(self.coordgraph.neighbors(aidx))
            for neigh in neighs:
                status = state[neigh][0]
                if status == 2:  # neighbor Fail
                    bonus += self.p_fail_bonus
                elif status == 3:  # neighbor dead
                    bonus += self.p_dead_bonus
            bonus /= len(neighs)
            p_fail = self.p_fail_base + bonus
            p_dead = self.p_dead_base + bonus

            # Rewards only if noop
            if action[aidx] == 0:  # noop
                status = state[aidx][0]
                if status == 1:  # Good
                    if random.random() < p_fail:
                        newstatus = 2
                    else:
                        newstatus = 1
                elif status == 2:
                    if random.random() < p_dead:
                        newstatus = 3
                    else:
                        newstatus = 2
                elif status == 3:
                    newstatus = 3

                load = state[aidx][1]
                if load == 2:  # work
                    if newstatus == 1:
                        if random.random() < self.p_doneG:
                            reward = 1.0  # finish reward
                    elif newstatus == 2:
                        if random.random() < self.p_doneF:
                            reward = 1.0  # finish reward
            else:  # reboot
                reward += self.reboot_penalty
            total_reward[aidx] = reward

        return total_reward

    def step(self, state, action):
        next_state = [None] * self.n_agents
        total_reward = [None] * self.n_agents
        for aidx in range(self.n_agents):
            reward = 0.0
            bonus = 0.0
            neighs = list(self.coordgraph.neighbors(aidx))
            for neigh in neighs:
                status = state[neigh][0]
                if status == 2:  # neighbor Fail
                    bonus += self.p_fail_bonus
                elif status == 3:  # neighbor dead
                    bonus += self.p_dead_bonus
            bonus /= len(neighs)
            p_fail = self.p_fail_base + bonus
            p_dead = self.p_dead_base + bonus

            # Rewards only if noop
            if action[aidx] == 0:  # noop
                status = state[aidx][0]
                if status == 1:  # Good
                    if random.random() < p_fail:
                        newstatus = 2
                    else:
                        newstatus = 1
                elif status == 2:
                    if random.random() < p_dead:
                        newstatus = 3
                    else:
                        newstatus = 2
                elif status == 3:
                    newstatus = 3

                load = state[aidx][1]
                if load == 1:  # idle
                    if newstatus == 1:
                        if random.random() < self.p_load:
                            newload = 2
                        else:
                            newload = 1
                    elif newstatus == 2:
                        if random.random() < self.p_load:
                            newload = 2
                        else:
                            newload = 1
                    elif newstatus == 3:
                        newload = 1
                elif load == 2:  # work
                    if newstatus == 1:
                        if random.random() < self.p_doneG:
                            newload = 3
                            reward = 1.0  # finish reward
                        else:
                            newload = 2
                    elif newstatus == 2:
                        if random.random() < self.p_doneF:
                            newload = 3
                            reward = 1.0  # finish reward
                        else:
                            newload = 2
                    elif newstatus == 3:  # dead
                        newload = 1
                elif load == 3:  # done
                    newload = 3
            else:  # reboot
                newstatus = 1  # Good
                newload = 1
                reward += self.reboot_penalty
            next_state[aidx] = [newstatus, newload]
            total_reward[aidx] = reward

        return next_state, total_reward, False

    def agent_actions(self, p, idx, s):
        return [0, 1]

    def action_to_int(self, a):
        return a

    def reset(self):

        self.state = [[1, 1]] * self.n_agents

        return self.state

    def print_network(self, coordgraph):
        nx.draw(coordgraph)
        plt.show()


if __name__ == '__main__':
    env = MultiAgentSysAdmin(4, 'UniSysAdmin')
    p, ns = env.get_successors(0, 0)
    pr = env.get_next_states_dist(0, 0)
    reward = env.get_reward_function(0, 0)
    print('ok')
