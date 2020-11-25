from batch import Batched
from coba.benchmarks import Benchmark
import re

def baseLearner():
    from coba.learners import VowpalLearner
    return VowpalLearner(seed=10, epsilon=0.1, flags='--coin')

learner_factories = [
    Batched(delay=8, batchsize=1, learner=baseLearner),
    Batched(delay=8, batchsize=2, learner=baseLearner),
    Batched(delay=8, batchsize=4, learner=baseLearner),
    Batched(delay=8, batchsize=8, learner=baseLearner),
]

processes = 4
maxtasksperchild = 1
json = "./exp.json"

log = re.sub('json$', 'log', json)

if __name__ == '__main__':
    result = Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).evaluate(learner_factories, log)
    result.standard_plot()
