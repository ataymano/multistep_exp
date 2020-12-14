from batch import Batched
from coba.benchmarks import Benchmark
import re

learner_factories = [
    Batched(delay=8, batchsize=1, seed=10, epsilon=0.1, flags='--coin'),
    Batched(delay=8, batchsize=2, seed=10, epsilon=0.1, flags='--coin'),
    Batched(delay=8, batchsize=4, seed=10, epsilon=0.1, flags='--coin'),
    Batched(delay=8, batchsize=8, seed=10, epsilon=0.1, flags='--coin'),
]

processes = 4
maxtasksperchild = 1
json = "./exp.json"

log = re.sub('json$', 'log', json)

if __name__ == '__main__':
    result = Benchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).evaluate(learner_factories, log)
    result.standard_plot()
