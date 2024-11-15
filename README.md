# Genetic Algorithm Optimization for Neural Networks

This project is a Python implementation of a Genetic Algorithm (GA) to optimize the architecture and hyperparameters of a Multi-Layer Perceptron (MLP) neural network.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Designing an optimal neural network architecture and hyperparameters can be a complex and time-consuming task. This project aims to automate this process by leveraging the power of Genetic Algorithms. The GA is used to evolve the number of layers, number of neurons per layer, activation functions, optimizer, and loss function of the MLP model in order to find the best performing configuration for a given task.

## Features
- Flexible configuration of GA parameters (population size, number of generations, etc.)
- Mutation and crossover operators for evolving the neural network architecture and hyperparameters
- Training and evaluation of the MLP models on a provided dataset
- Persistence of the best performing models across generations
- Visualization of the training and test set performance

## Installation
1. Clone the repository: https://github.com/CeIest2/HypParaOPtimiser.git

2. Navigate to the project directory: `cd HyperParamOptimiser`

3. Create a virtual environment, activate it and install the required dependencies:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Usage
1. Prepare your dataset and update the `Data_set` class in `objet_pour_AG.py` with the appropriate file paths and data preprocessing steps.
2. Customize the GA parameters in the `Para_AG` class in `objet_pour_AG.py` to fit your needs.
3. Run the `algo_genetique` function in the main script to start the optimization process.
4. Monitor the training progress and the performance of the best models.

## Project Structure
- `algo_genetique.py`: Main script to run the Genetic Algorithm optimization.
- `mutation.py`: Functions for mutating the neural network architecture and hyperparameters.
- `croisement.py`: Functions for crossover (mating) of individuals in the population.
- `train.py`: Functions for training and evaluating the neural network models.
- `selection.py`: Functions for selecting the best performing individuals in the population.
- `objet_pour_AG.py`: Defines the main classes used in the project (Genome, Individu, Population, Para_AG, Data_set).

## Contributing
Contributions to this project are welcome! If you find any issues or have ideas for improvements, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).


