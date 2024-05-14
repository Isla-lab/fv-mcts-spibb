import random
import time
import networkx as nx
from numba import prange, njit
import numpy as np
from agents.abstract_mcts import AbstractStateNode

_log_i = 1
_log_means = [0, 0]
_LOG_C = 1000


def ucb1(node: AbstractStateNode):
    """
    computes the best action based on the ucb1 value
    """
    n_visits = node.ns
    visit_child = []
    node_values = []
    # children = list(node.actions.values())
    children = sorted(node.actions.items())
    for c in children:
        visit_child.append(c[1].na)
        node_values.append(c[1].total / c[1].na)

    ucb_score = np.array(node_values) + node.param.C * np.sqrt(np.log(n_visits) / np.array(visit_child))

    # to avoid biases we randomly choose between the actions which maximize the ucb1 score
    index = np.random.choice(np.flatnonzero(ucb_score == ucb_score.max()))

    return node.actions[index].data


def discrete_default_policy(n_actions: int):
    """
    random policy
    :type n_actions: the number of available actions
    :return:
    """
    n_actions = n_actions

    def policy(*args, **kwargs):
        # choose an action uniformly random
        indices = list(range(n_actions))
        probs = [1 / len(indices)] * len(indices)
        sampled_action = random.choices(population=indices, weights=probs)[0]
        return sampled_action

    return policy


def grid_policy(prior_knowledge, n_actions):
    """
    a rollout policy based on
    :param prior_knowledge: a dictionary where the key is the state and the value is a vector
    representing the value of an action based on the knowledge (lower the value the better the action)
    :param n_actions: the number of available actions
    :return:
    """
    knowledge = prior_knowledge

    def policy(node: AbstractStateNode):
        """
        computes the best action based on the heuristic
        :return:
        """
        env = node.param.env
        state = env.__dict__["s"]
        ks = np.array(knowledge[state])

        # to avoid biases if two or more actions have the same value we choose randomly between them
        sampled_action = np.random.choice(np.flatnonzero(ks == ks.min()))

        return sampled_action

    return policy
