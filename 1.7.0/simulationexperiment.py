"""
This is an example script that benchmarks a vowpal wabbit bandit learner.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import coba.random

from coba.simulations import LambdaSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import Benchmark

import numpy as np

from batch import Batched
from coba.benchmarks import Benchmark
import re

def baseLearner():
    from coba.learners import VowpalLearner
    return VowpalLearner(seed=10, epsilon=0.2, flags='--coin')

def advantageLearner():
    from advantage import Advantage
    return Advantage(seed=10, flags='--coin', learner=baseLearner)

def get_context(means, t):
    return (str(t % means.shape[0]), str(coba.random.randint(0, means.shape[1] - 1)))

def get_actions(means):
    return [str(i) for i in range(means.shape[2])]

def get_reward(means, c, a):
    return int(coba.random.random() < means[int(c[0])][int(c[1])][int(a)]) 

if __name__ == '__main__':
    nsteps = 1
    npeople = 4
    nactions = 4

    means = np.ndarray(shape = (nsteps, npeople, nactions), buffer = np.array(coba.random.randoms(nsteps * npeople * nactions)))
    
    random_perf = np.mean(means)
    best_perf = np.max(means, axis=2).mean()

    print(f'Random perfomance: {random_perf}')
    print(f'Best performance: {best_perf}')
    epsilon = 0.2
    print(f'Best performance with {epsilon} exploration: {best_perf * (1 - epsilon) + random_perf * epsilon}')


    actions_objects = get_actions(means)

    contexts = lambda t: get_context(means, t)
    actions = lambda t: actions_objects

    rewards = lambda c, a: get_reward(means, c, a)

    #define a simulation
    simulations = [
        LambdaSimulation(10000, contexts, actions, rewards, seed=10),
    ]

    #define a benchmark: this benchmark replays the simulation 15 times
    benchmark = Benchmark(simulations, batch_size = 1, shuffle_seeds=list(range(5)))

    #create the learner factories
    learner_factories = [
        RandomLearner(seed=10),
        VowpalLearner(epsilon=0.2, seed=10),
    ]

    benchmark.evaluate(learner_factories).standard_plot()