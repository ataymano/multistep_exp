from batch import Batched
from coba.benchmarks import UniversalBenchmark, LearnerFactory
from coba.analysis import Plots
import re

learner_factories = [
    LearnerFactory(Batched, delay=8, batchsize=1, seed=10, epsilon=0.1, flags='--coin'),
    LearnerFactory(Batched, delay=8, batchsize=2, seed=10, epsilon=0.1, flags='--coin'),
    LearnerFactory(Batched, delay=8, batchsize=4, seed=10, epsilon=0.1, flags='--coin'),
    LearnerFactory(Batched, delay=8, batchsize=8, seed=10, epsilon=0.1, flags='--coin'),
]

processes = 4
maxtasksperchild = 1
json = "./exp.json"

log = re.sub('json$', 'log', json)

if __name__ == '__main__':
    result = UniversalBenchmark.from_file(json).processes(processes).maxtasksperchild(maxtasksperchild).evaluate(learner_factories, log)
    Plots.standard_plot(result)
