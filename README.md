# Metadata-based Multi-Task Bandits with Bayesian Hierarchical Models

This repository is the official implementation of the paper [Metadata-based Multi-Task Bandits with Bayesian Hierarchical Models]() in Python. 

>📋  **Abstract**: How to explore efficiently is a central problem in multi-armed bandits. In this paper, we introduce the metadata-based multi-task bandit problem, where the agent needs to solve a large number of related multi-armed bandits tasks and can leverage some task-specific features (i.e., metadata) to share knowledge across tasks. As a general framework, we propose to capture task relations through the lens of Bayesian hierarchical models, upon which a Thompson sampling algorithm is designed to efficiently learn task relations, share information, and minimize the cumulative regrets. Two concrete examples for Gaussian bandits and Bernoulli bandits are carefully analyzed. The Bayes regret for Gaussian bandits clearly demonstrates the benefits of information sharing with our algorithm. The proposed method is further supported by extensive experiments.

## Requirements

To install requirements:

```setup
conda env create -f MTS.yml
```
>📋  Note: The first line of the yml file can be altered to set the new environment's name (i.e., python3). To activate the new environment, please use the following line of code: conda activate python3.

## Functions for Numerical Experiments
### MTTS Algorithms and Other Baseline Algorithms
1. **Gaussian Bandits**: within the folder `/Gaussian`, there are codes for six different Thompson Sampling-based methods and a code to simulate data under the Gaussian bandits setting.
    1. `_agent_LB.py`: code to implement the Linear Bandits Algorithm. 
    2. `_agent_MTB.py`: code to implement the proposed algorithm MTTS. (Note: we use MTB and MTTS interchangeably)
    3. `_agent_TS.py`: code to implement basic Thompson Sampling Algorithms--OSFA(TS), individual-TS(N_TS), and oracle-TS(oracle).
    4. `_agent_meta_TS.py`: code to implement meta Thompson sampling algorithm with slight modification, refer to `https://arxiv.org/pdf/2102.06129.pdf`.
    5. `_env.py`: simulation environment of multi-task Gaussian bandits with baseline features.
2. **Bernoulli Bandits**: within the folder `/Binary`, there are codes for six different Thompson Sampling-based methods and a code to simulate data under the Bernoulli bandits setting.
    1. `_agent_GLB.py`: code to implement the Generalized Linear Bandits Algorithm. Refer to `https://arxiv.org/pdf/1906.08947.pdf`.
    2. `_agent_MTB_binary.py`: code to implement the proposed algorithm MTTS. (Note: we use MTB and MTTS interchangeably)
    3. `_agent_TS_binary.py`: code to implement basic Thompson Sampling Algorithms--OSFA(TS), individual-TS(N_TS), and oracle-TS(oracle).
    4. `_agent_meta_TS_binary.py`: code to implement meta Thompson sampling algorithm with slight modification, refer to `https://arxiv.org/pdf/2102.06129.pdf`.
    5. `_envBin.py`: simulation environment of multi-task binary bandits with baseline features.

### Other Functions Required
The following three code files, in the main folder, are used to conduct the experiments and get the results under both Gaussian bandits setting and Bernoulli bandits setting.
1. `_util.py`: helper functions.
2. `_experiement.py`: function to run the experiment under either the Gaussian bandits setting or the Bernoulli bandits setting.
3. `_analyzer.py`: post-process simulation results.

## Scripts to Conduct Experiments
Within the folder `/Experiment_Scripts`, there are four scripts for different experiments corresponding to results discussed in the paper and Appendix.
1. `Binary_Fig2.ipynb`: Script to reproduce the simulation results showed in **Figure2**(Section 6) and **Figure5**(Section E.2) under Binary bandits setting.
2. `Gaussian_Fig2.ipynb`: Script to reproduce the simulation results showed in **Figure2**(Section 6) and **Figure4**(Section E.2) under Gaussian bandits setting.
3. `Gaussian_Misspecify.ipynb`: Script to reproduce the simulation results showed in **Figure3**(Appendix E.1) under Gaussian bandits setting.
4. `Gaussian_KPN.ipynb`: Script to reproduce the simulation results showed in **Figure6**(Appendix E.3) under Gaussian bandits setting.

## Script to Analyze the Results
To generate the plots(figures) included in the paper, the following script is used.
1. `Results_Plot.ipynb`: script to reproduce the **Figure2--6**.

## Steps to Reproduce the Simulation Results
1. Install the required packages or create a new environment using MTS.yml (Refer to the **Requirements**); 
2. Download all the required codes in the same folder (Main Folder);
3. Within the Main Folder, create two empty folders `/res` and `/log` to save simulation results and create another empty folder `/fig` to save figures;
4. Run the corresponding experiment scripts to get the simulation results;
5. Analyze the results and get the figure by running the corresponding code in the `Results_Plot.ipynb`.
