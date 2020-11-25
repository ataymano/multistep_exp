from batch import Batched
from coba.benchmarks import Benchmark
import re

def baseLearner():
    from coba.learners import VowpalLearner
    return VowpalLearner(seed=10, epsilon=0.1, flags='--coin')

def advantageLearner():
    from advantage import Advantage
    return Advantage(seed=10, flags='--coin', learner=baseLearner)

learner_factories = [
    Batched(delay=8, batchsize=1, learner=advantageLearner),
    Batched(delay=8, batchsize=2, learner=advantageLearner),
    Batched(delay=8, batchsize=4, learner=advantageLearner),
    Batched(delay=8, batchsize=8, learner=advantageLearner),
]

processes = 4
maxtasksperchild = 1
json = "./exp.json"

log = re.sub('json$', 'log', json)
try:
    import os
    os.remove(log)
except:
    pass

if __name__ == '__main__':
    result = Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).evaluate(learner_factories, log)
    result.standard_plot()