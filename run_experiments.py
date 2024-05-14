import os
import sys
import configparser
import time

from experiment import MultiAgentSysAdminExperiment, MultiDroneDeliveryExperiment

directory = os.path.dirname(os.path.expanduser(__file__))
sys.path.append(directory)

path_config = configparser.ConfigParser()
path_config.read(os.path.join(directory, 'paths.ini'))
results_directory_absolute = path_config['PATHS']['results_path']

config_name = sys.argv[1]
experiment_config = configparser.ConfigParser()
experiment_config.read(os.path.join(directory, 'experiments', config_name))
experiment_directory_relative = experiment_config['META']['experiment_path_relative']
environment = experiment_config['META']['env_name']
machine_specific_directory = sys.argv[2]

experiment_directory = os.path.join(results_directory_absolute, experiment_directory_relative)
machine_specific_experiment_directory = os.path.join(experiment_directory, machine_specific_directory)

if not os.path.isdir(results_directory_absolute):
    os.mkdir(results_directory_absolute)
if not os.path.isdir(experiment_directory):
    os.mkdir(experiment_directory)
if not os.path.isdir(machine_specific_experiment_directory):
    os.mkdir(machine_specific_experiment_directory)

n_agents = experiment_config['ENV_PARAMETERS']['MACHINES']
nb_iterations = int(sys.argv[4])


def run_experiment(seed):

    if 'SysAdmin' in config_name:
        experiment = MultiAgentSysAdminExperiment(experiment_config=experiment_config, seed=seed, n_agents=n_agents,
                                                  nb_iterations=nb_iterations,
                                                  machine_specific_experiment_directory=
                                                  machine_specific_experiment_directory)
    elif 'Multi_Drone_Delivery' in config_name:
        experiment = MultiDroneDeliveryExperiment(experiment_config=experiment_config, seed=seed, n_agents=n_agents,
                                                  nb_iterations=nb_iterations,
                                                  machine_specific_experiment_directory=
                                                  machine_specific_experiment_directory)
    else:
        sys.exit('No domain selected')

    experiment.run()


if __name__ == '__main__':
    seed = int(sys.argv[3])
    f = open(os.path.join(machine_specific_experiment_directory, "Exp_description.txt"), "w+")
    f.write(f"This is Multi-agent SysAdmin/Multi-UAV Delivery benchmark used for {nb_iterations} iterations.\n")
    f.write(f'The seed used is {seed}.\n')
    f.write(f'Experiment starts at {time.ctime()}.')
    f.close()
    run_experiment(seed)
