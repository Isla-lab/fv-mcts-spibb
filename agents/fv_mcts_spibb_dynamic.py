import ast
import random
import time
from typing import Any
import collections
from envs.multiUAVDelivery import UAVGeneralState
import networkx as nx
import numpy as np
from agents.abstract_mcts import AbstractMcts, AbstractStateNode, AbstractActionNode
from agents.parameters.mcts_parameters import MctsParameters

param_mcts = None
param_spibb = None


# Dynamic version
def transition_factored_MLE(state, action):
    # If the coordination graph is not observed in data, then return the same state
    if state[-1] not in param_spibb.hist_coord_graph.keys():
        return state

    observation = []
    coordgraph = param_spibb.hist_coord_graph[state[-1]]
    for aidx in range(param_mcts.number_of_agents):
        s = [state[aidx]]
        for neigh in list(coordgraph.neighbors(aidx)):
            s.append(state[neigh])
        s.append(state[-1])

        if str(s) not in param_spibb.MLE_T[aidx].keys():
            observation.append(s[0])
        elif str(s) in param_spibb.MLE_T[aidx].keys() and \
                str(action[aidx]) not in param_spibb.MLE_T[aidx][str(s)].keys():
            observation.append(s[0])
        elif sum(list(param_spibb.MLE_T[aidx][str(s)][str(action[aidx])].values())) == 0:
            observation.append(s[0])
        else:
            ns = list(param_spibb.MLE_T[aidx][str(s)][str(action[aidx])].keys())
            p_ns = list(param_spibb.MLE_T[aidx][str(s)][str(action[aidx])].values())
            observation.append(eval(ns[np.random.choice(list(range(len(ns))), p=p_ns)]))

    ns_coordination_graph = param_mcts.env.coordination_graph(param_mcts.env.mdp, observation)
    hash = nx.weisfeiler_lehman_graph_hash(ns_coordination_graph)
    observation.append(hash)

    return observation


def get_boot_act(state_agent, agent):
    p = ~param_spibb.mask[agent][str(state_agent)] * param_spibb.pi_b.get_single_prob_of_baseline(state_agent, agent)
    p /= np.sum(p)

    return np.random.choice(list(range(param_mcts.env.n_actions_per_agent)), p=p)


def get_prob_nonboot(state_agent, agent):
    if str(state_agent) not in param_spibb.mask[agent].keys() or len(param_spibb.mask[agent][str(state_agent)]) == 0:
        return 0
    elif all(param_spibb.mask[agent][str(state_agent)]):
        return 1.0

    # p is a vector of probabilities in which non-bootstrapped actions have p != 0, bootstrapped actions have p = 0
    p = param_spibb.mask[agent][str(state_agent)] * param_spibb.pi_b.get_single_prob_of_baseline(state_agent, agent)

    return sum(p)


# Dynamic version with bootstrapped constraints (theoretically proved)
def max_plus_spibb(node: AbstractStateNode, node_exploration=False):
    # If the coordination graph is not observed in data, then return an action of the baseline
    if node.data[-1] not in param_spibb.hist_coord_graph.keys():
        return np.array(param_spibb.pi_b.get_act_from_baseline(node.data[:-1]))

    """ PART 1: Select bootstrapped action or not
    - Each agent selects independently an action. If it exists an agent that selects with the baseline, then all the 
    agents are constrained to select with the baseline
    """
    bootstrapped = False
    coordgraph = param_spibb.hist_coord_graph[node.data[-1]]
    action = np.full(node.param.number_of_agents, -1)
    # Each agent selects independently a bootstrapped action with the baseline or non-bootstrapped action with Max-Plus
    for aidx in range(node.param.number_of_agents):
        s = [node.data[aidx]]
        for neigh in list(coordgraph.neighbors(aidx)):
            s.append(node.data[neigh])
        s.append(node.data[-1])

        # Select a bootstrapped or non-bootstrapped action
        p_boot = 1 - get_prob_nonboot(s, aidx)
        if np.random.random() <= p_boot:  # select bootstrapped action
            bootstrapped = True

        action[aidx] = param_spibb.pi_b.get_single_act_from_baseline(s[0], aidx)

    if bootstrapped:
        return action

    "##################################################################################################################"
    """ PART 2: Message passing with constraints
    - Propagate the value of the bootstrapped actions for the agents that have already selected their actions.
    - Propagate the value of all the actions for the agents that have not selected their actions.
    """
    n_agents = node.param.number_of_agents
    temp = [param_mcts.env.agent_actions(param_mcts.env.mdp, a, node.data)
            for a in range(n_agents)]

    actions_per_agent = []
    for a in temp:
        actions_per_agent.append([param_mcts.env.action_to_int(i) for i in a])

    # # Limit the message passing for agents that selected bootstrapped actions
    # agents_that_use_baseline = np.where(action != -1)[0]
    # if agents_that_use_baseline.size != 0:
    #     for a in agents_that_use_baseline:
    #         actions_per_agent[a] = [action[a]]

    n_actions_values = param_mcts.env.n_actions_per_agent  # maximum
    if param_mcts.dynamic_coordination_graph:
        graph = node.coordgraph
    else:
        graph = node.param.coordination_graph
    edges = list(graph.edges)
    n_edges = len(edges)
    fwd_messages = np.zeros((n_actions_values, n_edges))
    bwd_messages = np.zeros_like(fwd_messages)
    # mu = np.zeros((n_actions_values, n_actions, n_actions))
    for t in range(node.param.message_passing_it):
        bwd_messages_old = bwd_messages.copy()
        fwd_messages_old = fwd_messages.copy()
        for iter, (i, j) in enumerate(edges):
            # Forward
            for aj in actions_per_agent[j]:
                fwd_values = list()
                for ai in actions_per_agent[i]:
                    val = node.Q[i][ai] - bwd_messages_old[ai, iter] + node.Qij[(i, j)][
                        ai, aj]  # + exploration_constant * sqrt( (log(state_total_n + 1.0)) / (state_stats.edge_action_n[e_idx, edge_tup_indices[ai_idx, aj_idx]] + 1) )
                    fwd_values.append(val)
                fwd_messages[aj][iter] = max(fwd_values)
            # Backward
            for ai in actions_per_agent[i]:
                bwd_values = list()
                for aj in actions_per_agent[j]:
                    val = node.Q[j][aj] - fwd_messages_old[aj, iter] + node.Qij[(i, j)][aj, ai]
                    bwd_values.append(val)
                bwd_messages[ai][iter] = max(bwd_values)
        # Normalization: quite involved, not ideal
        for col in range(fwd_messages.shape[1]):
            if sum(fwd_messages[:, col] != 0) != 0:
                fwd_messages[:, col] = np.where(fwd_messages[:, col] != 0, fwd_messages[:, col] - (
                        np.sum(fwd_messages[:, col]) / sum(fwd_messages[:, col] != 0)), 0)
        # Normalization: quite involved, not ideal
        for col in range(bwd_messages.shape[1]):
            if sum(bwd_messages[:, col] != 0) != 0:
                bwd_messages[:, col] = np.where(bwd_messages[:, col] != 0, bwd_messages[:, col] - (
                        np.sum(bwd_messages[:, col]) / sum(bwd_messages[:, col] != 0)), 0)

        fwd_norm = np.abs(fwd_messages - fwd_messages_old)
        bwd_norm = np.abs(bwd_messages - bwd_messages_old)

        temp_Q = node.Q.copy()
        for i in range(n_agents):
            temp_Q[i] /= n_agents

        for i in range(n_agents):
            for j in list(graph.neighbors(i)):
                try:
                    edge_index = edges.index((i, j))
                except:
                    edge_index = edges.index((j, i))
                if graph.has_edge(i, j):  # use backward message
                    temp_Q[i] += bwd_messages[:, edge_index]
                if graph.has_edge(j, i):
                    temp_Q[i] += fwd_messages[:, edge_index]
        if np.allclose(fwd_norm, np.zeros_like(fwd_norm)) and np.allclose(bwd_norm, np.zeros_like(bwd_norm)):
            break

    "##################################################################################################################"
    """ PART 3: Actions selection (only non-bootstrapped actions)
    - Agents that have not selected a bootstrapped actions, select independently a non-bootstrapped action
    - after M rounds of message passing
    """
    agents_that_use_max_plus = np.where(action == -1)[0]
    for i in agents_that_use_max_plus:
        Q_a = temp_Q[i].copy()
        if node_exploration:
            val = np.array(node.param.C * np.sqrt(
                np.divide(np.log((node.ns + 1)), node.N[i],
                          out=np.full_like(node.N[i], np.inf),
                          where=node.N[i] != 0)
            ))
            Q_a += val
        s = [node.data[i]]
        for neigh in list(coordgraph.neighbors(i)):
            s.append(node.data[neigh])
        s.append(node.data[-1])
        # Filter the actions that belong to the set of actions that agents can do and to the set of non-bootstrapped
        # actions
        # print(f'actions_per_agent {actions_per_agent}')
        # print(f'mask {str(s) in param_spibb.mask[i].keys()}')
        available_actions = np.array(list(set(np.where(param_spibb.mask[i][str(s)] == True)[0]) &
                                          set(actions_per_agent[i])))
        # print(f'available actions {available_actions}')
        # print(f'np.array(Q_a)[available_actions] {np.array(Q_a)[available_actions]}')
        # print(f'np.array(Q_a)[available_actions].max() {np.array(Q_a)[available_actions].max()}')
        # print(f'{np.flatnonzero(np.array(Q_a)[available_actions])}')
        # print(f'len Q_a= {len(Q_a)}')
        # available_actions = np.array(actions_per_agent)[i][np.where(param_spibb.mask[i][str(s)] == True)[0]]
        action[i] = available_actions[np.random.choice(np.flatnonzero(np.array(Q_a)[available_actions]
                                                                      == np.array(Q_a)[available_actions].max()))]

    return np.array(action)


class FV_MCTS_SPIBB_dynamic(AbstractMcts):
    """
    MonteCarlo Tree Search Safe Policy Improvement with Baseline Boostrapping (MCTS-SPIBB)
    """
    NAME = 'FV-MCTS_SPIBB_dynamic'

    def __init__(self, param: MctsParameters, param_spibb):
        super().__init__(param)
        self.state = param.initial_state
        self.n_sim = param.n_sim
        self.param_spibb = param_spibb

    def fit(self) -> int | np.ndarray:
        """
        Starting method, builds the tree and then gives back the best action

        :return: the best action
        """
        global param_mcts
        global param_spibb
        param_mcts = self.param
        param_spibb = self.param_spibb

        self.root = StateNodeHash(
            data=param_mcts.initial_state,
            param=param_mcts
        )
        for s in range(param_mcts.n_sim):
            self.root.build_tree_state(0)

        # print('\n\nFINAL:')
        a_star = max_plus_spibb(self.root, node_exploration=False)

        return a_star


class StateNodeHash(AbstractStateNode):
    def __init__(self, data: Any, param: MctsParameters):
        super().__init__(data, param)
        self.Q, self.Qij, self.N, self.Nij = self.initialize_graph_statistics(param.env)

    def get_prob_nonboot(self, state):
        if str(state) not in param_spibb.mask.keys():
            return 0
        elif param_spibb.mask[str(state)] == []:
            return 0
        temp = param_spibb.mask[str(state)] * param_spibb.pi_b.get_prob_of_baseline(state)
        temp = np.where(temp, 0, 1) + temp
        p_non_boot = 0
        for i in range(len(temp)):
            p_non_boot += np.prod(temp[i])

        return p_non_boot

    def build_tree_state(self, curr_depth):
        """
        go down the tree until a leaf is reached and do rollout from that
        :param curr_depth:  max depth of simulation
        :return:
        """
        # start = time.time()
        action = max_plus_spibb(self, node_exploration=True)
        # end = time.time()
        # print('Time Max-Plus %i' % (end - start))
        if str(action) in self.actions.keys():
            child = self.actions.get(str(action))
        else:
            child = ActionNodeHash(data=action, param=self.param)
            self.actions[str(action)] = child

        reward = child.build_tree_action(self.data, curr_depth)
        self.ns += 1
        self.update_stats(action, reward)
        self.total += reward

        return reward

    def initialize_graph_statistics(self, env):
        Q = dict()
        N = dict()
        Qij = dict()
        Nij = dict()

        for a in range(env.n_agents):
            Q[a] = np.zeros(env.n_actions_per_agent)
            N[a] = np.zeros(env.n_actions_per_agent)

        if not param_mcts.dynamic_coordination_graph:
            for e in env.coordgraph.edges:
                Qij[e] = np.zeros((env.n_actions_per_agent, env.n_actions_per_agent))
                Nij[e] = np.zeros((env.n_actions_per_agent, env.n_actions_per_agent))
        else:
            self.coordgraph = param_mcts.env.coordination_graph(param_mcts.env.mdp, self.data[:-1])
            for e in self.coordgraph.edges:
                Qij[e] = np.zeros((env.n_actions_per_agent, env.n_actions_per_agent))
                Nij[e] = np.zeros((env.n_actions_per_agent, env.n_actions_per_agent))
        return Q, Qij, N, Nij

    def update_stats(self, action, q):
        if not param_mcts.dynamic_coordination_graph:
            edges = param_mcts.env.coordgraph.edges
        else:
            edges = self.coordgraph.edges

        for i in range(param_mcts.env.n_agents):
            ai = action[i]
            self.N[i][ai] += 1
            self.Q[i][ai] += (q[i] - self.Q[i][ai]) / self.N[i][ai]

        for e in edges:
            i = e[0]
            j = e[1]
            ai = action[i]
            aj = action[j]
            self.Nij[e][ai, aj] += 1
            self.Nij[e][aj, ai] += 1
            qe = q[i] + q[j]
            self.Qij[e][ai, aj] += (qe - self.Qij[e][ai, aj]) / self.Nij[e][ai, aj]
            self.Qij[e][aj, ai] += (qe - self.Qij[e][aj, ai]) / self.Nij[e][aj, ai]

    # Dynamic version
    def rollout(self, obs, curr_depth: int):
        """
        Play out until max depth or a terminal state is reached

        :param curr_depth: max depth of simulation
        :return: reward obtained from the state
        """
        """
        Play out until max depth or a terminal state is reached

        :param curr_depth: max depth of simulation
        :return: reward obtained from the state
        """
        done = False
        reward = np.zeros(param_mcts.number_of_agents)
        state_values = []
        starting_depth = 0
        while not done and curr_depth + starting_depth < param_mcts.max_depth:
            # print('Step %s' % starting_depth)
            # start = time.time()

            # If the coordination graph is not observed in data, then return an action of the baseline
            if obs[-1] not in param_spibb.hist_coord_graph.keys():
                sampled_action = np.array(param_spibb.pi_b.get_act_from_baseline(obs[:-1]))

            else:
                sampled_action = np.full(param_mcts.number_of_agents, -1)
                coordgraph = param_spibb.hist_coord_graph[obs[-1]]
                for aidx in range(param_mcts.number_of_agents):
                    s = [obs[aidx]]
                    for neigh in list(coordgraph.neighbors(aidx)):
                        s.append(obs[neigh])
                    # Check if the state for the agent aidx has only bootstrapped actions, in this case use the baseline for aidx
                    if str(s) not in param_spibb.mask[aidx].keys():
                        sampled_action[aidx] = param_spibb.pi_b.get_single_act_from_baseline(s, aidx)
                        continue

                    # Select a bootstrapped or non-bootstrapped action
                    p_boot = 1 - get_prob_nonboot(s, aidx)

                    if np.random.random() <= p_boot:  # select bootstrapped action
                        sampled_action[aidx] = get_boot_act(s, aidx)
                        continue
                    else:  # select nonbootstrapped action
                        sampled_action[aidx] = np.random.choice(np.where(param_spibb.mask[aidx][str(s)] == True)[0])
                        continue

            rew = param_mcts.env.reward_function(obs, sampled_action)

            reward += np.array([r * pow(self.param.gamma, starting_depth) for r in rew])
            obs = transition_factored_MLE(obs, sampled_action)
            done = False
            state_values.append(obs)
            starting_depth += 1
        return reward


class ActionNodeHash(AbstractActionNode):

    def build_tree_action(self, state, curr_depth) -> float:
        """
        go down the tree until a leaf is reached and do rollout from that
        :param curr_depth:  max depth of simulation
        :return:
        """

        observation = transition_factored_MLE(state, self.data)

        instant_reward = param_mcts.env.reward_function(state, self.data)
        terminal = False

        if terminal:  # if the node is terminal back-propagate instant reward
            state = self.children.get(str(observation), None)
            # add terminal states for visualization
            if state is None:
                # add child node
                state = StateNodeHash(data=observation, param=self.param)
                state.terminal = True
                self.children[str(observation)] = state

            self.total += instant_reward
            for i in range(param_mcts.number_of_agents):
                state.N[i][int(self.data[i])] += 1
            state.ns += 1

            return instant_reward
        else:  # check if the node has been already visited
            state = self.children.get(str(observation), None)
            if state is None:  # never visited
                # add child node
                state = StateNodeHash(data=observation, param=self.param)
                self.children[str(observation)] = state
                # ROLLOUT
                delayed_reward = self.param.gamma * state.rollout(observation, curr_depth + 1)

                # BACK-PROPAGATION
                for i in range(param_mcts.number_of_agents):
                    state.N[i][int(self.data[i])] += 1
                state.ns += 1
                self.total += (instant_reward + delayed_reward)
                state.total += (instant_reward + delayed_reward)
                return instant_reward + delayed_reward
            else:  # visited, therefore, go deeper the tree
                delayed_reward = self.param.gamma * state.build_tree_state(curr_depth + 1)

                # # BACK-PROPAGATION
                self.total += (instant_reward + delayed_reward)
                for i in range(param_mcts.number_of_agents):
                    state.N[i][int(self.data[i])] += 1

                return instant_reward + delayed_reward
