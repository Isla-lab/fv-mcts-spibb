# Scalable Safe Policy Improvement for Factored Multi-Agent MDPs

This project can be used to reproduce the experiments presented in:

- Scalable Safe Policy Improvement for Factored Multi-Agent MDPs


# Prerequisites

The project is implemented in Python 3.10 and tested on Ubuntu 20.04.6 LTS (for the full list of requirements please refer to file requirements.txt)

# Usage

We include the following:

    Libraries of the following algorithms:
        - FV-MCTS-SPIBB with Max-Plus
        - FV-MCTS-SPIBB with Var-El
        - FV-MCTS with Max-Plus
        - FV-MCTS with Var-El
        - MCTS-SPIBB (from Castellini et. al, ICML 2023)

    Environments:
        Multi-agent SysAdmin
        Multi-UAV Delivery

1. In order to execute the code, set the path within the file paths.ini

2. To run the Multi-agent SysAdmin experiments:
- Set the parameters (e.g., number of trajectories, gamma, number of agents, N^ and the algorithms to test) within the the file "experiments/Multi_agent_SysAdmin.ini"
- Then launch the file run_experiments.py with the following parameters: Multi_agent_SysAdmin.ini multi_agent_sysadmin_results 1234 (1234 is the seed used)

3. To run the Multi-UAV Delivery experiments:
- Set the parameters (e.g., number of trajectories, gamma, number of agents, N^ and the algorithms to test) within the the file "experiments/Multi_Drone_Delivery.ini"
- Then launch the file run_experiments.py with the following parameters: Multi_Drone_Delivery.ini multi_drone_delivery_results 1234 (1234 is the seed used)


# Contributors

- Federico Bianchi (federico.bianchi@univr.it or federicobianchi501@gmail.com)
- Edoardo Zorzi (edoardo.zorzi@univr.it)
- Alberto Castellini (alberto.castellini@univr.it)
- Alessandro Farinelli (alessandro.farinelli@univr.it)

# Reference
If you use our code in your work, please kindly cite our paper:

# License

This project is GPLv3-licensed. 
