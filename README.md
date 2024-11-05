
# Multi-Agent Shipping Environment

This repository provides a JAX-based multi-agent reinforcement learning (MARL) environment designed for simulating a container shipping network. The environment and training pipeline are implemented in JAX and Flax, leveraging JAXMARL for efficient MARL experimentation with configurable port and agent parameters.

## Overview

In this environment, agents represent ships navigating between ports, delivering containers to target destinations. Ports are categorized by traffic volume, creating a realistic distribution of container loads. The environment’s goal is to train agents to optimize routing strategies that maximize container deliveries within a given timeframe.

Key components:
- **Multi-Agent Setup**: Multiple agents interact with ports, competing or collaborating to deliver containers.
- **Realistic Environment**: Ports are set with varying distances and container volumes to reflect real-world shipping dynamics.
- **JAXMARL**: Built on JAX for efficient simulation and fast MARL training, making it scalable for complex multi-agent scenarios.

## Environment Details

- **Ports**: Configured into three tiers based on container volume:
  - **Tier 1**: High-volume ports (~15 ports with 500,000 containers/month)
  - **Tier 2**: Medium-volume ports (~60 ports with 50,000 containers/month)
  - **Tier 3**: Low-volume ports (~25 ports with 5,000 containers/month)

- **Agents (Ships)**:
  - Each agent has a cargo capacity of up to 10,000 containers.
  - Can load/unload up to 2,000 containers per day.
  - Agents aim to maximize container deliveries within a 60-day period.

- **Observation Space**: Each agent observes its:
  - Location (at port or at sea)
  - Departure and destination port (one-hot encoded)
  - Journey status: days spent and days remaining
  - Cargo: an array representing containers for each target port
  - Port state: available containers when docked

- **Action Space**: Agents select a destination port. When at a port, agents unload containers specific to that port. During sea travel, agents cannot change destinations mid-journey.

- **Reward Structure**: Agents earn rewards by successfully delivering (unloading) containers at target ports, incentivizing efficient routing strategies.

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies

Run the provided installation script to set up the required environment and dependencies.

```bash
bash quick_install.sh
```

### 3. Configure the Environment

The environment is configured using `ippo_shipping.yaml`, a Hydra-compatible YAML file. Customize the parameters to suit different experimental setups:

```yaml
# Environment Parameters
ENV_NAME: "shipping"
ENV_KWARGS:
  num_ports: 100
  num_agents: 100
  high_volume_ratio: 0.15
  med_volume_ratio: 0.65
  low_volume_ratio: 0.20
  high_volume_containers: 500000
  med_volume_containers: 50000
  low_volume_containers: 5000
  max_distance: 30
  min_distance: 1
  max_cargo: 10000
  max_unloads_per_day: 2000
  max_steps: 60
```

This file defines:
- **Port and Container Configuration**: The number of ports, container volumes for each tier, and port-to-port distances.
- **Agent Parameters**: The number of agents, cargo capacity, and maximum loading/unloading rates.
- **Training Parameters**: Hyperparameters like learning rate, timesteps, and number of training epochs.

### 4. Training the Agents

Use `train.py` to train agents using the IPPO (Independent Proximal Policy Optimization) algorithm. Adjust the configurations in `ippo_shipping.yaml` as needed for specific training goals.

Run the training script:

```bash
python train.py --config-name ippo_shipping.yaml
```

During training, the agents interact in the environment and learn policies to maximize container deliveries through reinforcement. Training progress and logs are tracked with Weights & Biases (wandb) based on the configurations provided in the YAML file.

### 5. Visualization

The `container_visualizer.py` script enables you to observe agent actions, container distribution, and port interactions visually. Use it to evaluate the performance of policies and gain insights into agent behaviors within the environment.

### 6. Testing and Debugging

To verify the environment’s functionality and troubleshoot potential issues, use `test_env.py`:

```bash
python test_env.py
```

This script checks the core functions, ensuring they behave as expected and helping debug complex multi-agent interactions.

## Repository Structure

- **train.py**: Main training script using JAX and IPPO.
- **container_visualizer.py**: Visualization tool for observing agent and port dynamics.
- **ippo_shipping.yaml**: Configuration file for customizing environment, agent parameters, and training settings.
- **quick_install.sh**: Installation script for setting up dependencies.
- **test_env.py**: Testing and debugging tool for validating the environment.

## Technical Stack

This environment is built on:
- **JAX**: For high-performance computation and efficient multi-agent simulation.
- **Flax**: Neural network library used to build and optimize MARL policies.
- **Hydra**: Configuration management, allowing easy experimentation with different environment setups.
- **JAXMARL**: The backbone for managing multi-agent reinforcement learning processes.

## Usage Tips

- **Experiment with Configurations**: Adjust port tiers, distances, agent counts, and other parameters in `ippo_shipping.yaml` to test various logistics strategies.
- **Monitor Training**: Use wandb logging to keep track of model performance, reward progression, and training stability.
- **Scale Up with JAXMARL**: Leverage JAX’s performance benefits to increase the number of agents and ports for larger simulations.
