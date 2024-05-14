import random
from typing import Any
import collections
import numpy as np
from agents.abstract_mcts import AbstractMcts, AbstractStateNode, AbstractActionNode, AbstractUnfactoredStateNode
from agents.parameters.mcts_parameters import MctsParameters


exp_params = None
param_spibb = None
dict_of_actions = None

import itertools


def calculate_combinations(na, agents):
    num_range = range(na)
    all_combinations = itertools.product(num_range, repeat=agents)
    combinations_dict = {}

    for i, combo in enumerate(all_combinations):
        combinations_dict[np.array_str(np.array(combo)).replace(' ', ', ')] = i

    return combinations_dict


def ucb1_spibb(node: AbstractUnfactoredStateNode):
    """
    computes the best action based on the ucb1 value
    """
    n_visits = node.ns
    visit_child = []
    node_values = []
    children = list(node.actions_nonboot.values())
    for c in children:
        visit_child.append(c.na)
        node_values.append(c.total / c.na)

    ucb_score = np.array(node_values) + node.param.C * np.sqrt(
        np.divide(np.log(n_visits), np.array(visit_child))
    )

    # ucb_score = np.array(node_values) + node.param.C * np.sqrt(np.log(n_visits) / np.array(visit_child))

    # to avoid biases we randomly choose between the actions which maximize the ucb1 score
    # TODO here ucb_score could be np.array([]), i.e. empty
    index = np.random.choice(np.flatnonzero(ucb_score == ucb_score.max()))
    return children[index].data


class MCTS_SPIBB(AbstractMcts):
    """
    MonteCarlo Tree Search Safe Policy Improvement with Baseline Boostrapping (MCTS-SPIBB)
    """
    NAME = 'MCTS_SPIBB'

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
        global dict_of_actions
        param_mcts = self.param
        param_spibb = self.param_spibb
        dict_of_actions = calculate_combinations(param_mcts.env.n_actions_per_agent, param_mcts.number_of_agents)
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
            
            # `Real' choice
            a_pi_b = self.get_act_from_baseline(self.root.data)
            if str(self.root.data) not in param_spibb.non_boot_action.keys():
                return a_pi_b #boot_action
            elif str(a_pi_b) not in param_spibb.non_boot_action[str(self.root.data)]:
                return a_pi_b #boot_action
            else:
                q_values = np.array([x.total for x in self.root.actions_nonboot.values()])
                indexes = np.where(q_values == q_values.max())
                index = np.random.choice(indexes[0])
                a_star = list(self.root.actions_nonboot.keys())[index] #choose among the best non-boot action
                
                return a_star

        

    def get_act_from_baseline(self, state):
        p_baseline = param_spibb.pi_b.get_prob_of_baseline(state)

        joint_act = []
        for ai in range(param_mcts.number_of_agents):
            joint_act.append(np.random.choice(list(range(param_mcts.env.n_actions_per_agent)), p=p_baseline[ai]))

        return joint_act


class StateNodeHash(AbstractUnfactoredStateNode):
    def __init__(self, data: Any, param: MctsParameters):
        super().__init__(data, param)
        if str(data) in param_spibb.non_boot_action.keys():
            self.visited_actions_nonboot = np.zeros(len(param_spibb.non_boot_action[str(data)]))
        else:
            self.visited_actions_nonboot = []
        self.actions_boot = {}
        self.actions_nonboot = {}
        self.total = np.zeros(param.env.nb_actions)

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

    def get_boot_act(self, state, max_steps=10000):
        p_baseline = param_spibb.pi_b.get_prob_of_baseline(state)
        c = 0
        while True:
            joint_act = []
            for ai in range(param_mcts.number_of_agents):
                joint_act.append(np.random.choice([0, 1], p=p_baseline[ai]))

            if str(state) not in param_spibb.non_boot_action.keys():
                break
            elif str(joint_act) not in param_spibb.non_boot_action[str(state)] or c > max_steps:
                break
            c += 1

        return joint_act

    def build_tree_state(self, curr_depth):
        """
        go down the tree until a leaf is reached and do rollout from that
        :param curr_depth:  max depth of simulation
        :return:
        """

        p_boot = 1 - self.get_prob_nonboot(self.data)
        if np.random.random() <= p_boot:  # select bootstrapped action
            action = self.get_boot_act(self.data)
            if str(action) in self.actions_boot.keys():
                child = self.actions_boot.get(str(action))
            else:
                child = ActionNodeHash(data=action, param=self.param)
                self.actions_boot[str(action)] = child
        else:# select nonbootstrapped action
            if 0 in self.visited_actions_nonboot:
                idx_choice = np.random.choice(np.where(self.visited_actions_nonboot == 0)[0])
                action = list(param_spibb.non_boot_action[str(self.data)])[idx_choice]
                self.visited_actions_nonboot[idx_choice] = 1
            else:
                action = ucb1_spibb(self)
            if action in self.actions_nonboot.keys():
                child = self.actions_nonboot.get(action)
            else:
                child = ActionNodeHash(data=action, param=self.param)
                self.actions_nonboot[action] = child

        reward = child.build_tree_action(self.data, curr_depth)
        self.update_stats(action, np.sum(reward))

        return reward

    def update_stats(self, action, reward):
        self.ns += 1
        self.total[dict_of_actions[str(action)]] += reward

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
            p_boot = 1 - self.get_prob_nonboot(obs)
            if np.random.random() <= p_boot:  # select bootstrapped action
                sampled_action = self.get_boot_act(obs)
            else:  # select nonbootstrapped action
                sampled_action = np.random.choice(list(param_spibb.non_boot_action[str(obs)]))

            rew = param_mcts.env.reward_function(obs, sampled_action)
            # print(rew)

            reward += np.array([r * pow(self.param.gamma, starting_depth) for r in rew])

            if str(obs) not in param_spibb.MLE_T.keys():
                obs = obs
            elif str(obs) in param_spibb.MLE_T.keys() and \
                    str(sampled_action) not in param_spibb.MLE_T[str(obs)].keys():
                obs = obs
            elif sum(list(param_spibb.MLE_T[str(obs)][str(sampled_action)].values())) == 0:
                obs = obs
            else:
                ns = list(param_spibb.MLE_T[str(obs)][str(sampled_action)].keys())
                p_ns = list(param_spibb.MLE_T[str(obs)][str(sampled_action)].values())
                obs = eval(ns[np.random.choice(list(range(len(ns))), p=p_ns)])

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
        if str(state) not in param_spibb.MLE_T.keys():
            observation = state
        elif str(state) in param_spibb.MLE_T.keys() and \
                str(self.data) not in param_spibb.MLE_T[str(state)].keys():
            observation = state
        elif sum(list(param_spibb.MLE_T[str(state)][str(self.data)].values())) == 0:
            observation = state
        else:
            ns = list(param_spibb.MLE_T[str(state)][str(self.data)].keys())
            p_ns = list(param_spibb.MLE_T[str(state)][str(self.data)].values())
            observation = eval(ns[np.random.choice(list(range(len(ns))), p=p_ns)])

        instant_reward = param_mcts.env.reward_function(state, self.data)
        terminal = False

        # if the node is terminal back-propagate instant reward
        if terminal:
            # print('It\'s terminal')
            state = self.children.get(str(observation), None)
            # add terminal states for visualization
            if state is None:
                # add child node
                state = StateNodeHash(data=observation, param=self.param)
                state.terminal = True
                self.children[str(observation)] = state
            # ORIGINAL
            self.total += sum(instant_reward)
            self.na += 1
            # MODIFIED
            state.ns += 1
            return instant_reward
        else:
            # check if the node has been already visited
            state = self.children.get(str(observation), None)
            if state is None:
                # add child node
                state = StateNodeHash(data=observation, param=self.param)
                self.children[str(observation)] = state
                # ROLLOUT
                delayed_reward = self.param.gamma * state.rollout(observation, curr_depth + 1)

                # BACK-PROPAGATION
                self.na += 1
                state.ns += 1
                self.total += sum(instant_reward + delayed_reward)
                #print(state.total[dict_of_actions[str(self.data)]])
                state.total[dict_of_actions[str(self.data)]] += sum(instant_reward + delayed_reward)
                return instant_reward + delayed_reward
            else:
                # go deeper the tree
                delayed_reward = self.param.gamma * state.build_tree_state(curr_depth + 1)

                # # BACK-PROPAGATION
                self.total += sum(instant_reward + delayed_reward)
                self.na += 1
                return instant_reward + delayed_reward