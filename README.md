# Metadata-based Multi-Task Bandits with Bayesian Hierarchical Models

This repository is the official implementation of the paper [Metadata-based Multi-Task Bandits with Bayesian Hierarchical Models](https://arxiv.org) in Python. 

>📋  **Abstract**: How to explore efficiently is a central problem in multi-armed bandits. In this paper, we introduce the metadata-based multi-task bandit problem, where the agent needs to solve a large number of related multi-armed bandits tasks and can leverage some task-specific features (i.e., metadata) to share knowledge across tasks. As a general framework, we propose to capture task relations through the lens of Bayesian hierarchical models, upon which a Thompson sampling algorithm is designed to efficiently learn task relations, share information, and minimize the cumulative regrets. Two concrete examples for Gaussian bandits and Bernoulli bandits are carefully analyzed. The Bayes regret for Gaussian bandits clearly demonstrates the benefits of information sharing with our algorithm. The proposed method is further supported by extensive experiments.

## Requirements

To install requirements:

```setup
conda env create -f MTS.yml
```
>📋  Note: The first line of the yml file can be altered to set the new environment's name (i.e., python3). To activate the new environment, please use the following line of code: conda activate python3

## MTTS Algorithms and Other Baseline Algorithms
1. **Gaussian Bandits**: within the folder `/Gaussian`, there are codes for six different Thompson Sampling-based methods and a code to simulate data under the Gaussian bandits setting.
    1. `_agent_LB.py`: code to implement the Linear Bandits Algorithm. 
    2. `_agent_MTB.py`: code to implement the proposed algorithm MTTS. (Note: we use MTB and MTTS interchangeably)
    3. `_agent_TS.py`: code to implement basic Thompson Sampling Algorithms--OSFA(TS), individual-TS(N_TS), and oracle-TS(oracle)
    4. `_agent_meta_TS.py`: code to implement meta Thompson sampling algorithm with slight modification, refer to `https://arxiv.org/pdf/2102.06129.pdf`
    5. `_env.py`: simulation environment of multi-task Gaussian bandits with baseline features.
2. **Bernoulli Bandits**: within the folder `/Binary`, there are codes for six different Thompson Sampling-based methods and a code to simulate data under the Bernoulli bandits setting.
    1. `_agent_GLB.py`: code to implement the Generalized Linear Bandits Algorithm. Refer to `https://arxiv.org/pdf/1906.08947.pdf`
    2. `_agent_MTB_binary.py`: code to implement the proposed algorithm MTTS. (Note: we use MTB and MTTS interchangeably)
    3. `_agent_TS_binary.py`: code to implement basic Thompson Sampling Algorithms--OSFA(TS), individual-TS(N_TS), and oracle-TS(oracle)
    4. `_agent_meta_TS_binary.py`: code to implement meta Thompson sampling algorithm with slight modification, refer to `https://arxiv.org/pdf/2102.06129.pdf`
    5. `_envBin.py`: simulation environment of multi-task binary bandits with baseline features.

## Other Required Codes to Run the Simulation
In the folder, there are three code files which are used to conduct the siulation study under both Gaussian bandits setting and Bernoulli bandits setting.
    1. `_util.py`: helper functions
    2. `_experiement.py`: function to run the experiment under either the Gaussian bandits setting or the Bernoulli bandits setting.

## Scripts for the Simulation Study








    1. `_analyzer.py`: post-process simulation results
>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
