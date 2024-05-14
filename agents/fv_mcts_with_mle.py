import time
from typing import Any

import networkx as nx
import numpy as np
from agents.abstract_mcts import AbstractMcts, AbstractStateNode, AbstractActionNode
from agents.parameters.mcts_parameters import MctsParameters
from agents.var_el import FunctionBuilder

param_mcts = None
param_spibb = None

USE_MAX_PLUS = True


def transition_factored_MLE(state, action):
    observation = []
    for aidx in range(param_mcts.number_of_agents):
        s = [state[aidx]]
        for neigh in list(param_mcts.env.coordgraph.neighbors(aidx)):
            s.append(state[neigh])
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

    return observation


def max_plus(node: AbstractStateNode, node_exploration=False):
    n_actions = node.param.number_of_agents
    temp = [param_mcts.env.agent_actions(param_mcts.env.mdp, a, node.data)
            for a in range(n_actions)]
    actions_per_agent = []
    for a in temp:
        actions_per_agent.append([param_mcts.env.action_to_int(i) for i in a])
        # actions_per_agent.append(range(10))
    # checks = [True if len(a) == param_mcts.env.n_actions_per_agent else False for a in temp]

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

        for i in range(n_actions):
            for j in list(graph.neighbors(i)):
                try:
                    edge_index = edges.index((i, j))
                except:
                    edge_index = edges.index((j, i))
                if graph.has_edge(i, j):  # use backward message
                    node.Q[i] += bwd_messages[:, edge_index]
                if graph.has_edge(j, i):
                    node.Q[i] += fwd_messages[:, edge_index]
        if np.allclose(fwd_norm, np.zeros_like(fwd_norm)) and np.allclose(bwd_norm, np.zeros_like(bwd_norm)):
            break

    action = []
    for i in range(n_actions):
        Q_a = node.Q[i].copy()
        if node_exploration:
            val = np.array(node.param.C * np.sqrt(
                np.divide(np.log((node.ns + 1)), node.N[i],
                          out=np.full_like(node.N[i], np.inf),
                          where=node.N[i] != 0)
            ))
            Q_a += val
        # Find the max
        m = float("-inf")
        choices = []
        for a in actions_per_agent[i]:
            if Q_a[a] > m:
                choices = [a]
                m = Q_a[a]
            elif Q_a[a] == m:
                choices.append(a)

        if len(choices) > 0:
            key = np.random.randint(0, len(choices))
        else:
            key = 0
        action.append(choices[key])

    return np.array(action)


def var_el(node: AbstractStateNode, node_exploration=False):
    n_actions = node.param.number_of_agents
    temp = [param_mcts.env.agent_actions(param_mcts.env.mdp, a, node.data)
            for a in range(n_actions)]
    actions_per_agent = dict()
    for i, a in enumerate(temp):
        actions_per_agent[i] = [param_mcts.env.action_to_int(i) for i in a]

    if param_mcts.dynamic_coordination_graph:
        graph = node.coordgraph
    else:
        graph = node.param.coordination_graph
    edges = list(graph.edges)

    builder = FunctionBuilder(edges, range(node.param.number_of_agents), node.Qij, actions_per_agent, node,
                              exploration_bonus=node_exploration, heuristic=False)

    argmax = builder.create_max_function().compute_argmax()
    return np.array([argmax[a] for a in range(node.param.number_of_agents)])


class FV_MCTS_with_MLE(AbstractMcts):
    """
    MonteCarlo Tree Search
    """
    NAME = 'FV_MCTS_with_MLE'

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

        for state in [param_mcts.initial_state]:
            self.root = StateNodeHash(
                data=state,
                param=param_mcts
            )
            # t_0 = time.time()
            for s in range(param_mcts.n_sim):
                # print('Simulation number %s' % s)
                self.root.build_tree_state(0)
            # t_1 = time.time()
            # total_time = t_1 - t_0
            if USE_MAX_PLUS:
                a_star = max_plus(self.root, node_exploration=False)
            else:
                a_star = var_el(self.root, node_exploration=False)
            # print(a_star)
        return a_star


class StateNodeHash(AbstractStateNode):
    def __init__(self, data: Any, param: MctsParameters):
        super().__init__(data, param)
        self.coordgraph = None
        self.Q, self.Qij, self.N, self.Nij = self.initialize_graph_statistics(param.env)

    def build_tree_state(self, curr_depth):
        """
        go down the tree until a leaf is reached and do rollout from that
        :param curr_depth:  max depth of simulation
        :return:
        """
        if USE_MAX_PLUS:
            action = max_plus(self, node_exploration=True)
        else:
            action = var_el(self, node_exploration=True)
        # action = var_el(self, node_exploration=True) TODO change this line to use var_el
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
            sampled_action = np.random.randint(low=param_mcts.env.min_action, high=param_mcts.env.max_action + 1,
                                               size=param_mcts.number_of_agents)

            rew = param_mcts.env.reward_function(obs, sampled_action)

            reward += np.array([r * pow(self.param.gamma, starting_depth) for r in rew])

            obs = transition_factored_MLE(obs, sampled_action)

            done = False
            state_values.append(obs)
            starting_depth += 1

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
            self.coordgraph = env.coordination_graph(env.mdp, self.data)
            for e in self.coordgraph.edges:
                Qij[e] = np.zeros((env.n_actions_per_agent, env.n_actions_per_agent))
                Nij[e] = np.zeros((env.n_actions_per_agent, env.n_actions_per_agent))
        return Q, Qij, N, Nij


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
                # print('delayed reward %s' % delayed_reward)

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
