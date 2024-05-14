import ast
import time
import pickle
import gzip
import networkx as nx
import numpy as np
from agents.fv_mcts_with_mle import FV_MCTS_with_MLE
from agents.mcts_spibb import MCTS_SPIBB
from agents.parameters.mcts_parameters import MctsParameters, MctsSPIBBParameters
from agents.fv_mcts_spibb_dynamic import FV_MCTS_SPIBB_dynamic
from agents.fv_mcts_spibb_static import FV_MCTS_SPIBB_static
from agents.fv_mcts import FV_MCTS
from envs.baseline_policy import MultiAgentSysAdminGenerativeBaselinePolicy, MultiUAVDeliveryGenerativeBaselinePolicy
from envs.multiUAVDelivery import MultiUAVDelivery
from envs.multi_agent_sysAdmin import MultiAgentSysAdmin
from utils.action_selection_functions import ucb1, discrete_default_policy

# Translate the names from the algorithms to the class.
algorithm_name_dict = {FV_MCTS.NAME: FV_MCTS,
                       FV_MCTS_with_MLE.NAME: FV_MCTS_with_MLE,
                       FV_MCTS_SPIBB_static.NAME: FV_MCTS_SPIBB_static,
                       FV_MCTS_SPIBB_dynamic.NAME: FV_MCTS_SPIBB_dynamic,
                       MCTS_SPIBB.NAME: MCTS_SPIBB}

global args


def create_agent(key, param_mcts, param_mcts_spibb=None):
    param = MctsParameters(
        gamma=param_mcts[0],
        C=param_mcts[1],
        action_selection_fn=param_mcts[12],
        rollout_selection_fn=discrete_default_policy,
        max_depth=param_mcts[2],
        n_states=param_mcts[3],
        n_actions=param_mcts[4],
        initial_state=param_mcts[5],
        n_sim=param_mcts[6],
        env=param_mcts[7],
        message_passing_it=param_mcts[8],
        number_of_agents=param_mcts[9],
        coordination_graph=param_mcts[10],
        matrix=param_mcts[11],
        dynamic_coordination_graph=param_mcts[13]

    )

    if key in {FV_MCTS_SPIBB_dynamic.NAME}:
        param_spibb = MctsSPIBBParameters(
            pi_b=param_mcts_spibb[0],
            MLE_T=param_mcts_spibb[1],
            mask=param_mcts_spibb[2],
            non_boot_action=param_mcts_spibb[3],
            hist_coord_graph=param_mcts_spibb[4]
        )
        return FV_MCTS_SPIBB_dynamic(param, param_spibb)

    elif key in {FV_MCTS_SPIBB_static.NAME}:
        param_spibb = MctsSPIBBParameters(
            pi_b=param_mcts_spibb[0],
            MLE_T=param_mcts_spibb[1],
            mask=param_mcts_spibb[2],
            non_boot_action=param_mcts_spibb[3],
            hist_coord_graph=param_mcts_spibb[4]
        )
        return FV_MCTS_SPIBB_static(param, param_spibb)

    elif key in {MCTS_SPIBB.NAME}:
        param_spibb = MctsSPIBBParameters(
            pi_b=param_mcts_spibb[0],
            MLE_T=param_mcts_spibb[1],
            mask=param_mcts_spibb[2],
            non_boot_action=param_mcts_spibb[3],
            hist_coord_graph=param_mcts_spibb[4]
        )
        return MCTS_SPIBB(param, param_spibb)

    elif key in {FV_MCTS_with_MLE.NAME}:
        param_spibb = MctsSPIBBParameters(
            pi_b=param_mcts_spibb[0],
            MLE_T=param_mcts_spibb[1],
            mask=param_mcts_spibb[2],
            non_boot_action=param_mcts_spibb[3],
            hist_coord_graph=param_mcts_spibb[4]
        )
        return FV_MCTS_with_MLE(param, param_spibb)

    elif key in {FV_MCTS.NAME}:
        return FV_MCTS(param)


class Experiment:

    def __init__(self, experiment_config, seed, n_agents, nb_iterations, machine_specific_experiment_directory):
        """
        :param experiment_config: config file which describes the experiment, see, for example,
        experiments/SysAdmin_MCTS_SPIBB.ini
        :param seed: seed for this experiment
        :param nb_iterations: number of iterations of this experiment
        :param machine_specific_experiment_directory: the directory in which the results will be stored
        """
        self.seed = seed
        np.random.seed(seed)
        self.experiment_config = experiment_config
        self.machine_specific_experiment_directory = machine_specific_experiment_directory
        self.filename_header = f'results_{seed}'
        self.n_agents = int(n_agents)
        self.nb_iterations = nb_iterations
        self.algorithms_dict = ast.literal_eval(self.experiment_config['ALGORITHMS']['algorithms_dict'])
        self.filename_header = f'results_{seed}'
        self.nb_iterations = nb_iterations
        print(f'Initialising experiment with seed {seed} and {nb_iterations} iterations.')
        print(f'The machine_specific_experiment_directory is {self.machine_specific_experiment_directory}.')
        self._set_env_params()

    def run(self):
        """
        Runs the experiment.
        """
        pass

    def _set_env_params(self):
        pass

    def _run_algorithms(self):
        """
        Runs all algorithms for one data set.
        """
        pass


class MultiAgentSysAdminExperiment(Experiment):
    # Inherits from the base class Experiment to implement the SysAdmin experiment specifically.

    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the SysAdmin experiment.
        """
        self.episodic = False
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.number_of_machines = int(self.experiment_config['ENV_PARAMETERS']['MACHINES'])
        self.env = MultiAgentSysAdmin(self.n_agents, self.experiment_config['ENV_PARAMETERS']['TYPE_OF_ARCHITECTURE'])
        self.nb_states = self.env.nb_states
        self.nb_actions = self.env.nb_actions
        self.initial_state = self.env.reset()
        self.baseline_method = self.experiment_config['BASELINE']['method']
        self.fixed_params_exp_list = [self.number_of_machines, self.seed, self.gamma, self.baseline_method]

        self.number_trajectory = ast.literal_eval(self.experiment_config['BASELINE']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['BASELINE']['length_trajectory'])

    def run(self):
        """
        Runs the experiment.
        """
        self.pi_b = MultiAgentSysAdminGenerativeBaselinePolicy(env=self.env, gamma=self.gamma,
                                                               method=self.baseline_method)
        for iteration in range(self.nb_iterations):
            print("Iteration %s:" % iteration)
            for nt in self.number_trajectory:
                for lt in self.length_trajectory:
                    print(f'Starting with number trajectory {nt} out of {self.number_trajectory} and '
                          f'length_trajectory {lt} out of {self.length_trajectory}.')
                    self._run_one_iteration(nt, lt, save_data=True)
                    self._run_one_iteration(nt, lt, save_data=False)

    def _run_one_iteration(self, nt, lt, save_data=True):
        """
        Runs one iteration on the multi-agent SysAdmin benchmark
        """

        if save_data:
            self.generate_data(self.n_agents, nt, lt, self.env, self.pi_b, save_data=True)
            self.prepare_data_for_factored_model(nt, lt)
            if 'MCTS-SPIBB' in list(self.algorithms_dict.keys()):
                self.prepare_data_for_unfactored_model(nt, lt)
        else:
            print('Run algorithms')
            self._run_algorithms()

    def prepare_data_for_factored_model(self, nt, lt):
        print('Load dataset')
        raw_data = self.load_data(f'dataset/sysAdmin_n_agents_{self.n_agents}_nb_traj_{nt}_steps_{lt}.pickle')
        print('Manipulate dataset for FV-MCTS-SPIBB')
        data_fact, n_mask_fact = self.manipulate_data_factored(raw_data)
        print('Compute model for FV-MCTS-SPIBB')
        for N_wedge in self.algorithms_dict['FV-MCTS_SPIBB_static']['hyperparam']:
            self.compute_compressed_model_fact(data_fact, n_mask_fact, N_wedge, nt, lt)

    def prepare_data_for_unfactored_model(self, nt, lt):
        print('Load dataset')
        raw_data = self.load_data(f'dataset/sysAdmin_n_agents_{self.n_agents}_nb_traj_{nt}_steps_{lt}.pickle')
        print('Compute model for MCTS_SPIBB')
        for N_wedge in self.algorithms_dict['MCTS_SPIBB']['hyperparam']:
            self.compute_compressed_model_unfact(raw_data, N_wedge, nt, lt)

    def generate_data(self, number_of_agents, number_trajectories, max_steps, env, pi, save_data=False):
        print('Generate dataset using the baseline policy')
        trajectories = []
        for i in range(number_trajectories):
            # if i % 250 == 0:
            #     print(f'traj {i}')
            nb_steps = 0
            state = self.env.reset()
            is_done = False
            while nb_steps < max_steps and not is_done:
                p = pi.get_prob_of_baseline(state)
                action_choice = np.zeros(self.env.n_agents, dtype=int)
                for aidx in range(self.env.n_agents):
                    action_choice[aidx] = np.random.choice(p[aidx].shape[0], p=p[aidx])

                next_state, _, _ = env.step(state, action_choice)
                trajectories.append([state, action_choice, next_state])
                state = next_state
                nb_steps += 1
        if save_data:
            with gzip.open(f"dataset/sysAdmin_n_agents_{number_of_agents}_nb_traj_{number_trajectories}"
                           f"_steps_{max_steps}.pickle", "wb") as output_file:
                pickle.dump(trajectories, output_file)

    def load_data(self, path):
        with gzip.open(f"{path}", "rb") as input_file:
            raw_data = pickle.load(input_file)
        return raw_data

    def compute_compressed_model_unfact(self, raw_data, N_wedge, nt, lt):
        # Computing n
        n = dict()
        n_mask = dict()
        for episode in raw_data:
            action = np.array_str(episode[1]).replace(' ', ', ')
            state = str(episode[0])
            next_state = str(episode[2])
            if state not in n:
                n[state] = dict()
                n_mask[state] = dict()  # state
            if action not in n[state]:
                n[state][action] = dict()  # [state][action]
                n_mask[state][action] = 1
            else:
                n_mask[state][action] += 1
            if next_state not in n[state][action]:
                n[state][action][next_state] = 1  # [state][action][next_state]
            else:
                n[state][action][next_state] += 1  # [state][action][next_state]

        MLE_T = dict()
        mask = dict()
        non_boot_action = dict()
        ostates = np.array(list(n.keys()))
        ostates = np.sort(ostates)
        for state in ostates:
            if state not in mask:
                state = str(state)
                MLE_T[state] = dict()

            oactions = np.array(list(n[state].keys()))
            oactions = np.sort(oactions)
            for action in oactions:
                if n_mask[state][action] > N_wedge:
                    if state not in mask:
                        mask[state] = []
                        non_boot_action[state] = set()
                    non_boot_action[state].add(action)

                    matrix = np.zeros((self.n_agents, self.env.n_actions_per_agent), dtype=bool)
                    for i, ai in enumerate(eval(action)):
                        matrix[i][ai] = True
                    mask[state].append(matrix)

                # self.known_states_actions.append((state, action))
                MLE_T[state][action] = dict()
                onextstate = np.array(list(n[state][action].keys()))
                onextstate = np.sort(onextstate)
                sum_of_visits = sum(list(n[state][action].values()))
                for next_state in onextstate:
                    MLE_T[state][action][next_state] = (n[state][action][next_state] / sum_of_visits)
            if state in mask.keys():
                if mask[state] != []:
                    mask[state] = np.stack(mask[state], axis=0)

        # Computing the MLE model
        for state in MLE_T.keys():
            for action in MLE_T[state].keys():
                list_ns = MLE_T[state][action].keys()
                sum_p_dead = 0
                na = 0
                for ns in list_ns:
                    ns = str(ns)
                    if ns not in MLE_T.keys():
                        sum_p_dead += MLE_T[state][action][ns]
                        MLE_T[state][action][ns] = 0
                    else:
                        na += 1
                for ns in list_ns:
                    ns = str(ns)
                    if ns in MLE_T.keys():
                        MLE_T[state][action][ns] += sum_p_dead / na
                        MLE_T[state][action][ns] = MLE_T[state][action][ns]

        # Normalizing the probabilities
        for state in MLE_T.keys():
            for action in MLE_T[state].keys():
                p = np.sum(list(MLE_T[state][action].values()))
                for ns in MLE_T[state][action].keys():
                    ns = str(ns)
                    if p != 0:
                        MLE_T[state][action][ns] /= p
                    else:
                        MLE_T[state][action][ns] = 0

        del n
        del n_mask

        with gzip.open(f"dataset/sysAdmin_n_agents_{self.n_agents}_MLE_T_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(MLE_T, output_file)
        with gzip.open(f"dataset/sysAdmin_n_agents_{self.n_agents}_mask_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(mask, output_file)
        with gzip.open(f"dataset/sysAdmin_n_agents_{self.n_agents}_non_boot_action_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(non_boot_action, output_file)

    def load_unfact_model(self, path_MLE_T, path_mask, path_non_boot_action):
        with gzip.open(f"{path_MLE_T}", "rb") as input_file:
            MLE_T = pickle.load(input_file)
        with gzip.open(f"{path_mask}", "rb") as input_file:
            mask = pickle.load(input_file)
        with gzip.open(f"{path_non_boot_action}", "rb") as input_file:
            non_boot_action = pickle.load(input_file)
        return MLE_T, mask, non_boot_action

    # Factored states and actions
    def manipulate_data_factored(self, raw_data):
        """
        Generates a data batch for a non-episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """
        data = []
        n_mask = []
        for a in range(self.env.n_agents):
            data.append(dict())
            n_mask.append(dict())
        for i in raw_data:
            state = i[0]
            action_choice = i[1]
            next_state = i[2]

            for aidx in range(self.env.n_agents):
                s = [state[aidx]]
                ns = [next_state[aidx]]
                for neigh in list(self.env.coordgraph.neighbors(aidx)):
                    s.append(state[neigh])

                # Create a new state
                if str(s) not in data[aidx].keys():
                    data[aidx][str(s)] = dict()

                # Create a new action
                if str(action_choice[aidx]) not in data[aidx][str(s)].keys():
                    data[aidx][str(s)][str(action_choice[aidx])] = dict()

                # Create a new next state for the agent aidx
                if str(ns[0]) not in data[aidx][str(s)][str(action_choice[aidx])].keys():
                    data[aidx][str(s)][str(action_choice[aidx])][str(ns[0])] = 0
                data[aidx][str(s)][str(action_choice[aidx])][str(ns[0])] += 1

                # Create the counter for the mask
                if str(s) not in n_mask[aidx].keys():
                    n_mask[aidx][str(s)] = dict()
                if str(action_choice[aidx]) not in n_mask[aidx][str(s)]:
                    n_mask[aidx][str(s)][str(action_choice[aidx])] = 0
                n_mask[aidx][str(s)][str(action_choice[aidx])] += 1

        return data, n_mask

    # Factored states and actions
    def compute_compressed_model_fact(self, data_fact, n_mask_fact, N_wedge, nt, lt):
        MLE_T_fact = []
        mask_fact = []
        # self.non_boot_action = []
        for a in range(self.env.n_agents):
            MLE_T_fact.append(dict())
            mask_fact.append(dict())
            # self.non_boot_action.append(dict())

        for aidx in range(self.n_agents):
            ostates = np.array(list(data_fact[aidx].keys()))
            ostates = np.sort(ostates)
            for state in ostates:
                MLE_T_fact[aidx][state] = dict()

                oactions = np.array(list(data_fact[aidx][state].keys()))
                oactions = np.sort(oactions)
                for action in oactions:
                    if n_mask_fact[aidx][state][action] > N_wedge:
                        if state not in mask_fact[aidx].keys():
                            mask_fact[aidx][state] = np.zeros(len(self.env.agent_actions(
                                self.env.mdp, a, eval(state))), dtype=bool)
                            # self.non_boot_action[aidx][state] = set()
                        mask_fact[aidx][state][eval(action)] = True
                        # self.non_boot_action[aidx][state].add(action)

                    # self.known_states_actions.append((state, action))
                    MLE_T_fact[aidx][state][action] = dict()
                    sum_of_visits = 0
                    onextstate = np.array(list(data_fact[aidx][state][action].keys()))
                    onextstate = np.sort(onextstate)
                    sum_of_visits = sum(list(data_fact[aidx][state][action].values()))
                    for next_state in onextstate:
                        MLE_T_fact[aidx][state][action][next_state] = \
                            (data_fact[aidx][state][action][next_state] / sum_of_visits)

        with gzip.open(f"dataset/sysAdmin_{self.n_agents}_MLE_T_fact_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(MLE_T_fact, output_file)
        with gzip.open(f"dataset/sysAdmin_{self.n_agents}_mask_fact_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(mask_fact, output_file)

    def load_fact_model(self, path_MLE_T_fact, path_mask_fact):
        with gzip.open(f"{path_MLE_T_fact}", "rb") as input_file:
            MLE_T_fact = pickle.load(input_file)
        with gzip.open(f"{path_mask_fact}", "rb") as input_file:
            mask_fact = pickle.load(input_file)

        return MLE_T_fact, mask_fact

    def _run_algorithms(self):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {FV_MCTS_SPIBB_static.NAME}:
                self._run_fv_mcts_spibb(key)
            elif key in {FV_MCTS.NAME}:
                self._run_fv_mcts(key)
            elif key in {MCTS_SPIBB.NAME}:
                self._run_mcts_spibb(key)
            elif key in {FV_MCTS_with_MLE.NAME}:
                self._run_fv_mcts_with_mle(key)

    def _run_fv_mcts(self, key):
        """
        Runs MCTS
        """
        n_sim = 100
        c = 20
        max_depth = 20
        number_of_steps = 20
        message_passing_it = 8
        n_experiments = 1000
        rewards_discounted = []
        rewards_undiscounted = []
        times = []
        matrix_version = True
        dynamic_coordination_graph = False
        for t in range(n_experiments):
            total_discounted = []
            total_undiscounted = []
            state_mcts = self.env.reset()
            random_r = 0
            tt = []
            for step in range(number_of_steps):
                param_mcts = [self.gamma, c, max_depth, self.nb_states, self.nb_actions, state_mcts,
                              n_sim, self.env, message_passing_it, self.number_of_machines,
                              self.env.coordination_graph(), matrix_version, None, dynamic_coordination_graph]
                mcts = create_agent(key, param_mcts)
                t_0 = time.time()
                action = mcts.fit()
                t_1 = time.time()
                tt.append(t_1 - t_0)
                print('Time %s' % (t_1 - t_0))
                print('Selected action %s' % action)
                state_mcts, reward, terminal = self.env.step(state_mcts, action)
                print('Next State %s' % state_mcts)
                print('Reward %s' % sum(reward))
                total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                total_undiscounted.append(np.sum(reward))
            rewards_discounted.append(np.sum(total_discounted))
            rewards_undiscounted.append(np.sum(total_undiscounted))
            times.append(np.mean(tt))
            total_discounted = []
            state = self.env.reset()

        print(rewards_undiscounted)
        print(times)

        print(
            f"""┌ discounted\n│  mean = {round(np.mean(rewards_discounted), 5)}\n└  stdev = {round(np.std(rewards_discounted), 5)}""")
        print(
            f"""┌ undiscounted\n│  mean = {round(np.mean(rewards_undiscounted), 5)}\n└  stdev = {round(np.std(rewards_undiscounted), 5)}""")

    def get_prob_nonboot(self, state):
        if str(state) not in self.mask.keys():
            return 0
        elif self.mask[str(state)] == []:
            return 0
        temp = self.mask[str(state)] * self.pi_b.get_prob_of_baseline(state)
        temp = np.where(temp, 0, 1) + temp
        p_non_boot = 0
        for i in range(len(temp)):
            p_non_boot += np.prod(temp[i])

        return p_non_boot

    def _run_fv_mcts_spibb(self, key, zero_unseen=True):
        """
        Runs FV-MCTS-SPIBB for one data set, with all hyper-parameters and test on real environment.
        """
        print('FV-MCTS-SPIBB')
        n_sim = 100
        c = self.n_agents + 2
        max_depth = 20
        number_of_steps = 20
        message_passing_it = 40
        n_experiments = 1000
        rewards_discounted = []
        rewards_undiscounted = []
        pib_rewards_discounted = []
        pib_rewards_undiscounted = []
        times = []
        matrix_version = True
        dynamic_coordination_graph = False

        MLE_T_fact, mask_fact = self.load_fact_model(
            f"dataset/sysAdmin_{self.n_agents}_MLE_T_fact_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/sysAdmin_{self.n_agents}_mask_fact_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle")

        selected_actions = dict()
        selected_actions_pib = dict()
        for t in range(n_experiments):
            print('Experiment %s' % t)
            selected_actions[t] = []
            selected_actions_pib[t] = []
            # Evaluating baseline
            pib_total_discounted = []
            pib_total_undiscounted = []
            sel_act_pib = []
            state_pib = self.env.reset()
            for step in range(number_of_steps):
                p_baseline = self.pi_b.get_prob_of_baseline(state_pib)
                action = []
                for ai in range(self.n_agents):
                    action.append(np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline[ai]))
                sel_act_pib.append(action)
                state_pib, reward, terminal = self.env.step(state_pib, action)
                pib_total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                pib_total_undiscounted.append(np.sum(reward))
            pib_rewards_discounted.append(np.sum(pib_total_discounted))
            pib_rewards_undiscounted.append(np.sum(pib_total_undiscounted))
            selected_actions_pib[t].append(sel_act_pib)

            # Evaluating MCTS-SPIBB-with-Max-Plus
            total_discounted = []
            total_undiscounted = []
            state_mcts = self.env.reset()
            tt = []
            sel_act = []
            for step in range(number_of_steps):
                param_mcts = [self.gamma, c, max_depth, self.nb_states, self.nb_actions, state_mcts,
                              n_sim, self.env, message_passing_it, self.number_of_machines,
                              self.env.coordination_graph(), matrix_version, None, dynamic_coordination_graph]
                # Passing to MCTS_SPIBB param_mcts and param_spibb
                mcts_spibb = create_agent(key, param_mcts, [self.pi_b, MLE_T_fact, mask_fact, None, None])
                t_0 = time.time()
                action = mcts_spibb.fit()
                sel_act.append(action)
                t_1 = time.time()
                tt.append(t_1 - t_0)
                # print('Time %s' % (t_1 - t_0))
                # print('Selected action %s' % action)
                state_mcts, reward, terminal = self.env.step(state_mcts, action)
                # print('Reward %s' % sum(reward))
                # print('Next State %s' % state)

                total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                total_undiscounted.append(np.sum(reward))
            rewards_discounted.append(np.sum(total_discounted))
            rewards_undiscounted.append(np.sum(total_undiscounted))
            times.append(np.mean(tt))
            total_discounted = []
            selected_actions[t].append(sel_act)

        print(
            f"""┌ Behavior Policy:\n│  mean = {round(np.mean(pib_rewards_discounted), 5)}\n└  stdev = {round(np.std(pib_rewards_discounted), 5)}""")

        print(
            f"""┌ FV-MCTS-SPIBB\n│  mean = {round(np.mean(rewards_discounted), 5)}\n└  stdev = {round(np.std(rewards_discounted), 5)}""")

    def _run_fv_mcts_with_mle(self, key, zero_unseen=True):
        """
        Runs FV-MCTS with MLE for one data set, with all hyper-parameters and test on real environment.
        """
        print('FV-MCTS-with-MLE')
        n_sim = 100
        c = self.n_agents + 2
        max_depth = 20
        number_of_steps = 20
        message_passing_it = 8
        n_experiments = 1000
        rewards_discounted = []
        rewards_undiscounted = []
        pib_rewards_discounted = []
        pib_rewards_undiscounted = []
        times = []
        matrix_version = True
        dynamic_coordination_graph = False

        MLE_T_fact, _ = self.load_fact_model(
            f"dataset/sysAdmin_{self.n_agents}_MLE_T_fact_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/sysAdmin_{self.n_agents}_mask_fact_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle")

        selected_actions = dict()
        selected_actions_pib = dict()
        for t in range(n_experiments):
            print('Experiment %s' % t)
            selected_actions[t] = []
            selected_actions_pib[t] = []
            # Evaluating baseline
            pib_total_discounted = []
            pib_total_undiscounted = []
            sel_act_pib = []
            state_pib = self.env.reset()
            for step in range(number_of_steps):
                p_baseline = self.pi_b.get_prob_of_baseline(state_pib)
                action = []
                for ai in range(self.n_agents):
                    action.append(np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline[ai]))
                sel_act_pib.append(action)
                state_pib, reward, terminal = self.env.step(state_pib, action)
                pib_total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                pib_total_undiscounted.append(np.sum(reward))
            pib_rewards_discounted.append(np.sum(pib_total_discounted))
            pib_rewards_undiscounted.append(np.sum(pib_total_undiscounted))
            selected_actions_pib[t].append(sel_act_pib)

            # Evaluating MFV-MCTS-with-MLE
            total_discounted = []
            total_undiscounted = []
            state_mcts = self.env.reset()
            tt = []
            sel_act = []
            for step in range(number_of_steps):
                param_mcts = [self.gamma, c, max_depth, self.nb_states, self.nb_actions, state_mcts,
                              n_sim, self.env, message_passing_it, self.number_of_machines,
                              self.env.coordination_graph(), matrix_version, None, dynamic_coordination_graph]
                # Passing to MCTS_SPIBB param_mcts and param_spibb
                fv_mcts_with_mle = create_agent(key, param_mcts, [None, MLE_T_fact, None, None, None])
                t_0 = time.time()
                action = fv_mcts_with_mle.fit()
                sel_act.append(action)
                t_1 = time.time()
                tt.append(t_1 - t_0)
                # print('Time %s' % (t_1 - t_0))
                # print('Selected action %s' % action)
                state_mcts, reward, terminal = self.env.step(state_mcts, action)
                # print('Reward %s' % sum(reward))
                # print('Next State %s' % state)

                total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                total_undiscounted.append(np.sum(reward))
            rewards_discounted.append(np.sum(total_discounted))
            rewards_undiscounted.append(np.sum(total_undiscounted))
            times.append(np.mean(tt))
            total_discounted = []
            selected_actions[t].append(sel_act)

        print(
            f"""┌ Behavior Policy: \n│  mean = {round(np.mean(pib_rewards_discounted), 5)}\n└  stdev = {round(np.std(pib_rewards_discounted), 5)}""")

        print(
            f"""┌ FV-MCTS: \n│  mean = {round(np.mean(rewards_discounted), 5)}\n└  stdev = {round(np.std(rewards_discounted), 5)}""")

    def _run_mcts_spibb(self, key, zero_unseen=True):
        """
        Runs MCTS-SPIBB for one data set, with all hyper-parameters.
        """
        print('MCTS-SPIBB')
        n_sim = 1000
        c = self.n_agents + 2
        max_depth = 20
        number_of_steps = 20
        n_experiments = 1000
        rewards_discounted = []
        rewards_undiscounted = []
        pib_rewards_discounted = []
        pib_rewards_undiscounted = []
        times = []
        matrix_version = True
        dynamic_coordination_graph = False

        MLE_T, mask, non_boot_action = self.load_unfact_model(
            f"dataset/sysAdmin_n_agents_{self.n_agents}_MLE_T_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/sysAdmin_n_agents_{self.n_agents}_mask_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/sysAdmin_n_agents_{self.n_agents}_non_boot_action_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle")

        selected_actions = dict()
        selected_actions_pib = dict()
        for t in range(n_experiments):
            print('Experiment %s' % t)
            selected_actions[t] = []
            selected_actions_pib[t] = []
            # Evaluating baseline
            pib_total_discounted = []
            pib_total_undiscounted = []
            sel_act_pib = []
            state_pib = self.env.reset()
            for step in range(number_of_steps):
                p_baseline = self.pi_b.get_prob_of_baseline(state_pib)
                action = []
                for ai in range(self.n_agents):
                    action.append(np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline[ai]))
                sel_act_pib.append(action)
                state_pib, reward, terminal = self.env.step(state_pib, action)
                pib_total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                pib_total_undiscounted.append(np.sum(reward))
            pib_rewards_discounted.append(np.sum(pib_total_discounted))
            pib_rewards_undiscounted.append(np.sum(pib_total_undiscounted))
            selected_actions_pib[t].append(sel_act_pib)

            # Evaluating MCTS-SPIBB
            total_discounted = []
            total_undiscounted = []
            state_mcts = self.env.reset()
            tt = []
            sel_act = []
            for step in range(number_of_steps):
                param_mcts = [self.gamma, c, max_depth, self.nb_states, self.nb_actions, state_mcts,
                              n_sim, self.env, None, self.number_of_machines,
                              self.env.coordination_graph(), None, None, None]
                # Passing to MCTS_SPIBB param_mcts and param_spibb
                mcts_spibb = create_agent(key, param_mcts, [self.pi_b, MLE_T, mask, non_boot_action,
                                                            None])
                t_0 = time.time()
                action = mcts_spibb.fit()
                sel_act.append(action)
                t_1 = time.time()

                tt.append(t_1 - t_0)
                print(t_1 - t_0)
                # print('Time %s' % (t_1 - t_0))
                # print('Selected action %s' % action)
                state_mcts, reward, terminal = self.env.step(state_mcts, action)
                # print('Reward %s' % sum(reward))
                # print('Next State %s' % state)

                total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                total_undiscounted.append(np.sum(reward))
            rewards_discounted.append(np.sum(total_discounted))
            rewards_undiscounted.append(np.sum(total_undiscounted))
            times.append(np.mean(tt))
            total_discounted = []
            selected_actions[t].append(sel_act)

        print(
            f"""┌ Behavior Policy: \n│  mean = {round(np.mean(pib_rewards_discounted), 5)}\n└  stdev = {round(np.std(pib_rewards_discounted), 5)}""")

        print(
            f"""┌ MCTS-SPIBB: \n│  mean = {round(np.mean(rewards_discounted), 5)}\n└  stdev = {round(np.std(rewards_discounted), 5)}""")


class MultiDroneDeliveryExperiment(Experiment):
    # Inherits from the base class Experiment to implement the Multi Drone Delivery experiment specifically.

    def _set_env_params(self):
        """
        Reads in all parameters necessary from self.experiment_config to set up the Multi Drone Delivery experiment.
        """
        self.episodic = False
        self.gamma = float(self.experiment_config['ENV_PARAMETERS']['GAMMA'])
        self.n_agents = int(self.experiment_config['ENV_PARAMETERS']['MACHINES'])
        self.baseline_method = self.experiment_config['BASELINE']['method']
        self.number_trajectory = ast.literal_eval(self.experiment_config['BASELINE']['number_trajectory'])
        self.length_trajectory = ast.literal_eval(self.experiment_config['BASELINE']['length_trajectory'])

        self.PSET = {
            8: {"XY_AXIS_RES": 0.2, "XYDOT_LIM": 0.2, "XYDOT_STEP": 0.2, "NOISESTD": 0.1},
            16: {"XY_AXIS_RES": 0.1, "XYDOT_LIM": 0.1, "XYDOT_STEP": 0.1, "NOISESTD": 0.05},
            32: {"XY_AXIS_RES": 0.08, "XYDOT_LIM": 0.08, "XYDOT_STEP": 0.08, "NOISESTD": 0.05},
            48: {"XY_AXIS_RES": 0.05, "XYDOT_LIM": 0.05, "XYDOT_STEP": 0.05, "NOISESTD": 0.02}
        }
        # Reward = goal_bonus, prox_scaling, repul_pen, dynamics_scaling
        self.rewset = (1000.0, 1.0, 10.0, 10.0)
        self.env = MultiUAVDelivery(self.n_agents, self.PSET, self.rewset)
        self.initial_state = self.env.reset()

    def run(self):
        """
        Runs the experiment.
        """
        self.pi_b = MultiUAVDeliveryGenerativeBaselinePolicy(self.n_agents, self.env)

        for iteration in range(self.nb_iterations):
            print("Iteration %s:" % iteration)
            for nt in self.number_trajectory:
                for lt in self.length_trajectory:
                    print(f'Starting with number trajectory {nt} out of {self.number_trajectory} and '
                          f'length_trajectory {lt} out of {self.length_trajectory}.')
                    self._run_one_iteration(nt, lt, save_data=True)
                    self._run_one_iteration(nt, lt, save_data=False)

    def _run_one_iteration(self, nt, lt, save_data=False):
        """
        Runs one iteration on the Multi Drone Delivery benchmark
        """

        if save_data:
            self.generate_data(self.n_agents, nt, lt, self.env, self.pi_b, save_data=True)
            self.prepare_data_for_factored_model(nt, lt)
            # self.prepare_data_for_unfactored_model(nt, lt)
            pass

        print('Run algorithms')
        self._run_algorithms()

    def generate_data(self, number_of_agents, number_trajectories, max_steps, env, pi, save_data=False):
        print('Generate dataset using the baseline policy')
        trajectories = []
        for i in range(number_trajectories):
            # if i % 250 == 0:
            #     print(f'traj {i}')
            nb_steps = 0
            state = self.initial_state.copy()
            is_done = False
            while nb_steps < max_steps and not is_done:
                p = pi.get_prob_of_baseline(state)
                action_choice = np.zeros(self.env.n_agents, dtype=int)
                for aidx in range(self.env.n_agents):
                    action_choice[aidx] = np.random.choice(p[aidx].shape[0], p=p[aidx])

                next_state, _, _ = env.step(env.mdp, state, action_choice)
                trajectories.append([state, action_choice, next_state])
                state = next_state.copy()
                nb_steps += 1
        if save_data:
            with gzip.open(f"dataset/multi_uav_n_agents_{number_of_agents}_nb_traj_{number_trajectories}"
                           f"_steps_{max_steps}.pickle", "wb") as output_file:
                pickle.dump(trajectories, output_file)

    def load_data(self, path):
        with gzip.open(f"{path}", "rb") as input_file:
            raw_data = pickle.load(input_file)
        return raw_data

    def prepare_data_for_factored_model(self, nt, lt):
        print('Load dataset')
        raw_data = self.load_data(f'dataset/multi_uav_n_agents_{self.n_agents}_nb_traj_{nt}_steps_{lt}.pickle')
        print('Manipulate dataset for factored version')
        data_fact, n_mask_fact, hist_coord_graph = self.manipulate_data_factored(raw_data)
        print('Compute model for factored version')
        for N_wedge in self.algorithms_dict['FV-MCTS_SPIBB_dynamic']['hyperparam']:
            self.compute_compressed_model_fact(data_fact, n_mask_fact, hist_coord_graph, N_wedge, nt, lt)

    def prepare_data_for_unfactored_model(self, nt, lt):
        print('Load dataset')
        raw_data = self.load_data(f'dataset/multi_uav_n_agents_{self.n_agents}_nb_traj_{nt}_steps_{lt}.pickle')
        print('Compute model for unfactored version')
        for N_wedge in self.algorithms_dict['MCTS_SPIBB']['hyperparam']:
            self.compute_compressed_model_unfact(raw_data, N_wedge, nt, lt)

    # Manipulate raw data
    def manipulate_data_factored(self, raw_data):
        """
        Generates a data batch for a non-episodic MDP.
        :param nb_steps: number of steps in the data batch
        :param env: environment to be used to generate the batch on
        :param pi: policy to be used to generate the data as numpy array with shape (nb_states, nb_actions)
        :return: data batch as a list of sublists of the form [state, action, next_state, reward]
        """

        data = []
        n_mask = []
        hist_coord_graph = dict()
        for a in range(self.env.n_agents):
            data.append(dict())
            n_mask.append(dict())
        for i in raw_data:
            state = i[0]
            action_choice = i[1]
            next_state = i[2]

            coordgraph = self.env.coordination_graph(self.env.mdp, state)
            # sorted_nodes = sorted(coordgraph.nodes())
            # coordgraph = coordgraph.subgraph(sorted_nodes).copy()
            hash = nx.weisfeiler_lehman_graph_hash(coordgraph)
            if hash not in hist_coord_graph.keys():
                hist_coord_graph[hash] = coordgraph

            for aidx in range(self.env.n_agents):
                s = [state[aidx]]
                ns = [next_state[aidx]]
                for neigh in list(coordgraph.neighbors(aidx)):
                    s.append(state[neigh])
                s.append(hash)

                # Create a new state
                if str(s) not in data[aidx].keys():
                    data[aidx][str(s)] = dict()

                # Create a new action
                if str(action_choice[aidx]) not in data[aidx][str(s)].keys():
                    data[aidx][str(s)][str(action_choice[aidx])] = dict()

                # Create a new next state for the agent aidx
                if str(ns[0]) not in data[aidx][str(s)][str(action_choice[aidx])].keys():
                    data[aidx][str(s)][str(action_choice[aidx])][str(ns[0])] = 0
                data[aidx][str(s)][str(action_choice[aidx])][str(ns[0])] += 1

                # Create the counter for the mask
                if str(s) not in n_mask[aidx].keys():
                    n_mask[aidx][str(s)] = dict()
                if str(action_choice[aidx]) not in n_mask[aidx][str(s)]:
                    n_mask[aidx][str(s)][str(action_choice[aidx])] = 0
                n_mask[aidx][str(s)][str(action_choice[aidx])] += 1

        return data, n_mask, hist_coord_graph

    # Factored states and actions
    def compute_compressed_model_fact(self, data_fact, n_mask_fact, hist_coord_graph, N_wedge, nt, lt):
        MLE_T_fact = []
        mask_fact = []
        for a in range(self.env.n_agents):
            MLE_T_fact.append(dict())
            mask_fact.append(dict())

        for aidx in range(self.n_agents):
            ostates = np.array(list(data_fact[aidx].keys()))
            ostates = np.sort(ostates)
            for state in ostates:
                MLE_T_fact[aidx][state] = dict()

                oactions = np.array(list(data_fact[aidx][state].keys()))
                oactions = np.sort(oactions)
                for action in oactions:
                    if n_mask_fact[aidx][state][action] > N_wedge:
                        if state not in mask_fact[aidx].keys():
                            mask_fact[aidx][state] = np.zeros(self.env.n_actions_per_agent, dtype=bool)
                        mask_fact[aidx][state][eval(action)] = True

                    MLE_T_fact[aidx][state][action] = dict()
                    sum_of_visits = 0
                    onextstate = np.array(list(data_fact[aidx][state][action].keys()))
                    onextstate = np.sort(onextstate)
                    sum_of_visits = sum(list(data_fact[aidx][state][action].values()))
                    for next_state in onextstate:
                        MLE_T_fact[aidx][state][action][next_state] = \
                            (data_fact[aidx][state][action][next_state] / sum_of_visits)

        with gzip.open(f"dataset/multi_uav_{self.n_agents}_MLE_T_fact_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(MLE_T_fact, output_file)
        with gzip.open(f"dataset/multi_uav_{self.n_agents}_mask_fact_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(mask_fact, output_file)
        with gzip.open(f"dataset/multi_uav_{self.n_agents}_hist_coord_graph_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(hist_coord_graph, output_file)

    def load_fact_model(self, path_MLE_T_fact, path_mask_fact, path_hist_coord_graph):
        with gzip.open(f"{path_MLE_T_fact}", "rb") as input_file:
            MLE_T_fact = pickle.load(input_file)
        with gzip.open(f"{path_mask_fact}", "rb") as input_file:
            mask_fact = pickle.load(input_file)
        with gzip.open(f"{path_hist_coord_graph}", "rb") as input_file:
            hist_coord_graph = pickle.load(input_file)

        return MLE_T_fact, mask_fact, hist_coord_graph

    def compute_compressed_model_unfact(self, raw_data, N_wedge, nt, lt):
        # Computing n
        n = dict()
        n_mask = dict()
        for episode in raw_data:
            action = np.array_str(episode[1]).replace(' ', ', ')
            state = str(episode[0])
            next_state = str(episode[2])
            if state not in n:
                n[state] = dict()
                n_mask[state] = dict()  # state
            if action not in n[state]:
                n[state][action] = dict()  # [state][action]
                n_mask[state][action] = 1
            else:
                n_mask[state][action] += 1
            if next_state not in n[state][action]:
                n[state][action][next_state] = 1  # [state][action][next_state]
            else:
                n[state][action][next_state] += 1  # [state][action][next_state]

        MLE_T = dict()
        mask = dict()
        non_boot_action = dict()
        ostates = np.array(list(n.keys()))
        ostates = np.sort(ostates)
        for state in ostates:
            if state not in mask:
                state = str(state)
                MLE_T[state] = dict()

            oactions = np.array(list(n[state].keys()))
            oactions = np.sort(oactions)
            for action in oactions:
                if n_mask[state][action] > N_wedge:
                    if state not in mask:
                        mask[state] = []
                        non_boot_action[state] = set()
                    non_boot_action[state].add(action)

                    matrix = np.zeros((self.n_agents, self.env.n_actions_per_agent), dtype=bool)
                    for i, ai in enumerate(eval(action)):
                        matrix[i][ai] = True
                    mask[state].append(matrix)

                # self.known_states_actions.append((state, action))
                MLE_T[state][action] = dict()
                onextstate = np.array(list(n[state][action].keys()))
                onextstate = np.sort(onextstate)
                sum_of_visits = sum(list(n[state][action].values()))
                for next_state in onextstate:
                    MLE_T[state][action][next_state] = (n[state][action][next_state] / sum_of_visits)
            if state in mask.keys():
                if mask[state] != []:
                    mask[state] = np.stack(mask[state], axis=0)

        # Computing the MLE model
        for state in MLE_T.keys():
            for action in MLE_T[state].keys():
                list_ns = MLE_T[state][action].keys()
                sum_p_dead = 0
                na = 0
                for ns in list_ns:
                    ns = str(ns)
                    if ns not in MLE_T.keys():
                        sum_p_dead += MLE_T[state][action][ns]
                        MLE_T[state][action][ns] = 0
                    else:
                        na += 1
                for ns in list_ns:
                    ns = str(ns)
                    if ns in MLE_T.keys():
                        MLE_T[state][action][ns] += sum_p_dead / na
                        MLE_T[state][action][ns] = MLE_T[state][action][ns]

        # Normalizing the probabilities
        for state in MLE_T.keys():
            for action in MLE_T[state].keys():
                p = np.sum(list(MLE_T[state][action].values()))
                for ns in MLE_T[state][action].keys():
                    ns = str(ns)
                    if p != 0:
                        MLE_T[state][action][ns] /= p
                    else:
                        MLE_T[state][action][ns] = 0

        del n
        del n_mask

        with gzip.open(f"dataset/multi_uav_{self.n_agents}_MLE_T_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(MLE_T, output_file)
        with gzip.open(f"dataset/multi_uav_{self.n_agents}_mask_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(mask, output_file)
        with gzip.open(f"dataset/multi_uav_{self.n_agents}_non_boot_action_nb_traj_{nt}"
                       f"_steps_{lt}.pickle", "wb") as output_file:
            pickle.dump(non_boot_action, output_file)

    def load_unfact_model(self, path_MLE_T, path_mask, path_non_boot_action):
        with gzip.open(f"{path_MLE_T}", "rb") as input_file:
            MLE_T = pickle.load(input_file)
        with gzip.open(f"{path_mask}", "rb") as input_file:
            mask = pickle.load(input_file)
        with gzip.open(f"{path_non_boot_action}", "rb") as input_file:
            non_boot_action = pickle.load(input_file)
        return MLE_T, mask, non_boot_action

    def _run_algorithms(self):
        """
        Runs all algorithms for one data set.
        """
        for key in self.algorithms_dict.keys():
            if key in {FV_MCTS_SPIBB_dynamic.NAME}:
                self._run_fv_mcts_spibb(key)
            elif key in {FV_MCTS.NAME}:
                self._run_fv_mcts(key)
            elif key in {MCTS_SPIBB.NAME}:
                self._run_mcts_spibb(key)
            elif key in {FV_MCTS_with_MLE.NAME}:
                self._run_fv_mcts_with_mle(key)

    def _run_fv_mcts(self, key):
        """
        Runs MCTS
        """
        n_sim = 100
        c = 5
        max_depth = 10
        message_passing_it = 8
        n_experiments = 1000
        rewards_discounted = []
        rewards_undiscounted = []
        times = []
        matrix_version = True
        dynamic_coordination_graph = True
        for t in range(n_experiments):
            total_discounted = []
            total_undiscounted = []
            state_mcts = self.initial_state.copy()
            random_r = 0
            tt = []
            print('Initial state: %s' % [a.coords for a in self.env.s0])
            for step in range(150):
                print('Goal regions: %s' % [i.cen for i in self.env.mdp.goal_regions])
                print('Goal regions capacity: %s' % [i.cap for i in self.env.mdp.goal_regions])
                print('Regions to UAV: %s' % self.env.mdp.region_to_uavids)

                param_mcts = [self.gamma, c, max_depth, None, None, state_mcts,
                              n_sim, self.env, message_passing_it, self.n_agents,
                              None, matrix_version, None, dynamic_coordination_graph]
                mcts = create_agent(key, param_mcts)
                t_0 = time.time()
                action = mcts.fit()
                t_1 = time.time()
                tt.append(t_1 - t_0)
                print('Time %s' % (t_1 - t_0))
                print('Selected action %s' % action)
                state_mcts, reward, terminal = self.env.step(self.env.mdp, state_mcts, action)
                print('Next State %s' % [a.coords for a in state_mcts])
                print('Agents in their goal regions? %s' %
                      [self.env.is_in_region(self.env.mdp.dynamics.params,
                                             self.env.mdp.goal_regions[self.env.mdp.uav_to_region_map[a]],
                                             state_mcts[a].coords)
                       for a in range(len(state_mcts))])
                # print('Sum of reward %s' % sum(reward))
                print('Reward %s' % reward)
                total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                total_undiscounted.append(np.sum(reward))
            rewards_discounted.append(np.sum(total_discounted))
            rewards_undiscounted.append(np.sum(total_undiscounted))
            times.append(np.mean(tt))
            total_discounted = []

        print(
            f"""┌ discounted\n│  mean = {round(np.mean(rewards_discounted), 5)}\n└  stdev = {round(np.std(rewards_discounted), 5)}""")
        print(
            f"""┌ undiscounted\n│  mean = {round(np.mean(rewards_undiscounted), 5)}\n└  stdev = {round(np.std(rewards_undiscounted), 5)}""")

    def _run_fv_mcts_spibb(self, key, zero_unseen=True):
        """
        Runs FV-MCTS-SPIBB for one data set, with all hyper-parameters and test on real environment.
        """
        print('FV-MCTS-SPIBB')
        n_sim = 100
        c = 5
        max_depth = 20
        number_of_steps = 100
        message_passing_it = 8
        n_experiments = 1000
        rewards_discounted = []
        rewards_undiscounted = []
        pib_rewards_discounted = []
        pib_rewards_undiscounted = []
        times = []
        matrix_version = True
        dynamic_coordination_graph = True

        target = self.n_agents

        MLE_T_fact, mask_fact, hist_coord_graph = self.load_fact_model(
            f"dataset/multi_uav_{self.n_agents}_MLE_T_fact_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/multi_uav_{self.n_agents}_mask_fact_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/multi_uav_{self.n_agents}_hist_coord_graph_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle"
        )

        selected_actions = dict()
        selected_actions_pib = dict()
        for t in range(n_experiments):
            print('Experiment %s' % t)
            selected_actions[t] = []
            selected_actions_pib[t] = []

            # Evaluating MCTS-SPIBB-with-Max-Plus
            total_discounted = []
            total_undiscounted = []

            state_mcts = self.initial_state.copy()
            coordgraph = self.env.coordination_graph(self.env.mdp, state_mcts)
            state_mcts.append(nx.weisfeiler_lehman_graph_hash(coordgraph))
            tt = []
            sel_act = []
            # for step in range(number_of_steps):
            step = 0
            action = [0] * self.n_agents
            while step <= number_of_steps:
                print(f'Step: {step}')
                # print('Goal regions: %s' % [i.cen for i in self.env.mdp.goal_regions])
                # print('Goal regions capacity: %s' % [i.cap for i in self.env.mdp.goal_regions])
                # print('Regions to UAV: %s' % self.env.mdp.region_to_uavids)
                param_mcts = [self.gamma, c, max_depth, None, None, state_mcts,
                              n_sim, self.env, message_passing_it, self.n_agents,
                              None, matrix_version, None, dynamic_coordination_graph]
                # Passing to MCTS_SPIBB param_mcts and param_spibb
                fv_mcts_spibb = create_agent(key, param_mcts, [self.pi_b, MLE_T_fact, mask_fact, None,
                                                               hist_coord_graph])
                t_0 = time.time()
                action = fv_mcts_spibb.fit()
                sel_act.append(action)
                t_1 = time.time()
                tt.append(t_1 - t_0)
                print('Time %s' % (t_1 - t_0))
                print('Selected action %s' % action)
                state_mcts, reward, terminal = self.env.step(self.env.mdp, state_mcts[:-1], action)
                coordgraph = self.env.coordination_graph(self.env.mdp, state_mcts)
                state_mcts.append(nx.weisfeiler_lehman_graph_hash(coordgraph))
                print('Reward %s' % sum(reward))
                print('Next State %s' % state_mcts)
                # print('Agents in their goal regions? %s' %
                #       [self.env.is_in_region(self.env.mdp.dynamics.params,
                #                              self.env.mdp.goal_regions[self.env.mdp.uav_to_region_map[a]],
                #                              state_mcts[a].coords)
                #        for a in range(len(state_mcts[:-1]))])
                total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                total_undiscounted.append(np.sum(reward))

                step += 1
                print('\n\n')
            rewards_discounted.append(np.sum(total_discounted))
            rewards_undiscounted.append(np.sum(total_undiscounted))
            times.append(np.mean(tt))
            total_discounted = []
            selected_actions[t].append(sel_act)

            # Evaluating baseline
            pib_total_discounted = []
            pib_total_undiscounted = []
            sel_act_pib = []
            state_pib = self.initial_state.copy()
            step = 0
            action = [0] * self.n_agents
            # for step in range(number_of_steps):
            while step <= number_of_steps:
                print(f'Step: {step}')

                p_baseline = self.pi_b.get_prob_of_baseline(state_pib)
                action = []
                for ai in range(self.n_agents):
                    action.append(np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline[ai]))
                sel_act_pib.append(action)
                state_pib, reward, terminal = self.env.step(self.env.mdp, state_pib, action)
                pib_total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                pib_total_undiscounted.append(np.sum(reward))
                if step % 2000 == 0:
                    print(pib_total_undiscounted)
                step += 1

            pib_rewards_discounted.append(np.sum(pib_total_discounted))
            pib_rewards_undiscounted.append(np.sum(pib_total_undiscounted))
            selected_actions_pib[t].append(sel_act_pib)

        print(
            f"""┌ Behavior Policy: \n│  mean = {round(np.mean(pib_rewards_discounted), 5)}\n└  stdev = {round(np.std(pib_rewards_discounted), 5)}""")


        print(
            f"""┌ FV-MCTS-SPIBB: \n│  mean = {round(np.mean(rewards_discounted), 5)}\n└  stdev = {round(np.std(rewards_discounted), 5)}""")


    def _run_fv_mcts_with_mle(self, key, zero_unseen=True):
        """
        Runs FV-MCTS with MLE for one data set, with all hyper-parameters and test on real environment.
        """
        print('FV-MCTS-with-MLE')
        n_sim = 100
        c = self.n_agents + 2
        max_depth = 10
        number_of_steps = 1
        message_passing_it = 8
        n_experiments = 1000
        rewards_discounted = []
        rewards_undiscounted = []
        pib_rewards_discounted = []
        pib_rewards_undiscounted = []
        times = []
        matrix_version = True
        dynamic_coordination_graph = True
        MLE_T_fact, _, hist_coord_graph = self.load_fact_model(
            f"dataset/multi_uav_{self.n_agents}_MLE_T_fact_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/multi_uav_{self.n_agents}_mask_fact_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/multi_uav_{self.n_agents}_hist_coord_graph_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle"
        )

        selected_actions = dict()
        selected_actions_pib = dict()
        for t in range(n_experiments):
            print('Experiment %s' % t)
            selected_actions[t] = []
            selected_actions_pib[t] = []
            # Evaluating baseline
            pib_total_discounted = []
            pib_total_undiscounted = []
            sel_act_pib = []
            state_pib = self.initial_state.copy()
            for step in range(number_of_steps):
                p_baseline = self.pi_b.get_prob_of_baseline(state_pib)
                action = []
                for ai in range(self.n_agents):
                    action.append(np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline[ai]))
                sel_act_pib.append(action)
                state_pib, reward, terminal = self.env.step(self.env.mdp, state_pib, action)
                pib_total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                pib_total_undiscounted.append(np.sum(reward))
            pib_rewards_discounted.append(np.sum(pib_total_discounted))
            pib_rewards_undiscounted.append(np.sum(pib_total_undiscounted))
            selected_actions_pib[t].append(sel_act_pib)

            # Evaluating MFV-MCTS-with-MLE
            total_discounted = []
            total_undiscounted = []
            state_mcts = self.initial_state.copy()
            coordgraph = self.env.coordination_graph(self.env.mdp, state_mcts)
            state_mcts.append(nx.weisfeiler_lehman_graph_hash(coordgraph))
            tt = []
            sel_act = []
            for step in range(number_of_steps):
                param_mcts = [self.gamma, c, max_depth, None, None, state_mcts,
                              n_sim, self.env, message_passing_it, self.n_agents,
                              None, matrix_version, None, dynamic_coordination_graph]
                # Passing to MCTS_SPIBB param_mcts and param_spibb
                fv_mcts_with_mle = create_agent(key, param_mcts, [None, MLE_T_fact, None, None,
                                                                  hist_coord_graph])
                t_0 = time.time()
                action = fv_mcts_with_mle.fit()
                sel_act.append(action)
                t_1 = time.time()
                tt.append(t_1 - t_0)
                print('Time %s' % (t_1 - t_0))
                print('Selected action %s' % action)
                state_mcts, reward, terminal = self.env.step(self.env.mdp, state_mcts[:-1], action)
                coordgraph = self.env.coordination_graph(self.env.mdp, state_mcts)
                state_mcts.append(nx.weisfeiler_lehman_graph_hash(coordgraph))
                print('Reward %s' % sum(reward))
                print('Next State %s' % state_mcts)

                total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                total_undiscounted.append(np.sum(reward))
            rewards_discounted.append(np.sum(total_discounted))
            rewards_undiscounted.append(np.sum(total_undiscounted))
            times.append(np.mean(tt))
            total_discounted = []
            selected_actions[t].append(sel_act)

        print(
            f"""┌ Behavior Policy: \n│  mean = {round(np.mean(pib_rewards_discounted), 5)}\n└  stdev = {round(np.std(pib_rewards_discounted), 5)}""")

        print(
            f"""┌ FV-MCTS: \n│  mean = {round(np.mean(rewards_discounted), 5)}\n└  stdev = {round(np.std(rewards_discounted), 5)}""")


    def _run_mcts_spibb(self, key, zero_unseen=True):
        """
        Runs MCTS-SPIBB for one data set, with all hyper-parameters.
        """
        n_sim = 100
        c = 20
        max_depth = 10
        number_of_steps = 1
        n_experiments = 1000
        rewards_discounted = []
        rewards_undiscounted = []
        pib_rewards_discounted = []
        pib_rewards_undiscounted = []
        times = []

        MLE_T, mask, non_boot_action = self.load_unfact_model(
            f"dataset/multi_uav_{self.n_agents}_MLE_T_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/multi_uav_{self.n_agents}_mask_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle",
            f"dataset/multi_uav_{self.n_agents}_non_boot_action_nb_traj_{self.number_trajectory[0]}"
            f"_steps_{self.length_trajectory[0]}.pickle")

        for t in range(n_experiments):
            # Evaluating baseline
            pib_total_discounted = []
            pib_total_undiscounted = []
            state_pib = self.initial_state.copy()
            for step in range(number_of_steps):
                p_baseline = self.pi_b.get_prob_of_baseline(state_pib)
                action = []
                for ai in range(self.n_agents):
                    action.append(np.random.choice(list(range(self.env.n_actions_per_agent)), p=p_baseline[ai]))
                state_pib, reward, terminal = self.env.step(self.env.mdp, state_pib, action)
                pib_total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                pib_total_undiscounted.append(np.sum(reward))
            pib_rewards_discounted.append(np.sum(pib_total_discounted))
            pib_rewards_undiscounted.append(np.sum(pib_total_undiscounted))

            # Evaluating MCTS-SPIBB
            total_discounted = []
            total_undiscounted = []
            state_mcts = self.initial_state.copy()
            tt = []
            for step in range(number_of_steps):
                param_mcts = [self.gamma, c, max_depth, self.nb_states, self.nb_actions, state_mcts,
                              n_sim, self.env, None, self.number_of_machines,
                              self.env.coordination_graph(), None, None, None]
                # Passing to MCTS_SPIBB param_mcts and param_spibb
                mcts_spibb = create_agent(key, param_mcts, [self.pi_b, MLE_T, mask, non_boot_action,
                                                            None])
                t_0 = time.time()
                action = mcts_spibb.fit()
                t_1 = time.time()
                tt.append(t_1 - t_0)
                print('Time %s' % (t_1 - t_0))
                print('Selected action %s' % action)
                state_mcts, reward, terminal = self.env.step(self.env.mdp, state_mcts, action)
                print('Reward %s' % sum(reward))
                print('Next State %s' % state_mcts)

                total_discounted.append(np.sum(reward) * pow(self.gamma, step))
                total_undiscounted.append(np.sum(reward))
            rewards_discounted.append(np.sum(total_discounted))
            rewards_undiscounted.append(np.sum(total_undiscounted))
            times.append(np.mean(tt))
            total_discounted = []

        print(
            f"""┌ Behavior Policy: \n│  mean = {round(np.mean(pib_rewards_discounted), 5)}\n└  stdev = {round(np.std(pib_rewards_discounted), 5)}""")


        print(
            f"""┌ MCTS-SPIBB\n│  mean = {round(np.mean(rewards_discounted), 5)}\n└  stdev = {round(np.std(rewards_discounted), 5)}""")

