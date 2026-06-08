# Reinforcement Learning for Earthquake Mitigation in a Two-Block Fault System

This repository contains a reinforcement learning (RL) project investigating whether an intelligent agent can mitigate large instability events in a simplified earthquake fault model.

## Overview

Many physical systems exhibit nonlinear dynamics, delayed effects of actions, and long-term consequences of control decisions. These properties make them challenging sequential decision-making problems.

In this project, a reinforcement learning agent interacts with a two-block spring-slider fault system and learns pressure-control policies that influence the fault dynamics. The objective is to reduce large slip events and improve overall system stability.

## Research Motivation

My broader research interests lie in reinforcement learning, representation learning, and intelligent decision-making in complex physical environments.

This project serves as an initial exploration of how learning-based agents can interact with and control nonlinear physical systems under uncertainty.

## Repository Structure

* `two_block_pressure_env.py` – RL environment implementation based on a two-block fault model
* `train_ppo.py` – PPO training script
* `numerical_integration.py` – Numerical integration utilities for system simulation
* `performance_comparison.py` – Comparison between learned policies and baseline behaviour
* `env_test.py` / `test.py` – Environment testing and validation scripts
* `environment.yml` – Conda environment configuration

## Current Status

This is an ongoing research project.

Current functionality includes:

* Two-block fault system simulation
* Pressure-based control actions
* PPO-based policy learning
* Baseline and policy performance comparison

Future updates may include:

* Improved documentation
* Additional experimental results
* World-model-based approaches
* Representation learning for physical systems
* More realistic fault dynamics

## Requirements

The project environment can be created using:

```bash
conda env create -f environment.yml
```

## Disclaimer

This repository is intended for research and educational purposes. The underlying fault model is a simplified dynamical system and does not represent a realistic earthquake forecasting or hazard mitigation tool.
