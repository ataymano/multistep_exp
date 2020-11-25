from coba.learners import VowpalLearner
from collections import defaultdict
import numpy as np

from typing import Hashable, Sequence, Dict, Any

class Batched:
    def __init__(self, delay: int, batchsize: int, epsilon: float, seed: int, flags: str):
        self.learner = VowpalLearner(seed=seed, epsilon=epsilon, flags=flags)
        self.batchsize = batchsize
        self.delay = delay
        self.epsilon = epsilon
        self.seed = seed
        self.flags = flags
        self.mem = {}

        assert self.delay % self.batchsize == 0

    @property
    def family(self) -> str:
        return "Batched CB"

    @property
    def params(self) -> Dict[str,Any]:
        return {'e':self.epsilon, 'seed': self.seed, 'batchsize': self.batchsize, 'delay': self.delay, 'flags': self.flags }

    def choose(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> int:
        """Choose which action index to take."""

        self.mem[key] = { 'context': context }

        return self.learner.choose(key, context, actions)

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float) -> None:
        """Learn about the result of an action that was taken in a context."""

        self.mem[key]['action'] = action
        self.mem[key]['reward'] = reward

        if len(self.mem) >= self.delay:
            sumreward = 0
            contexts = []
            for key, values in self.mem.items():
                sumreward += values['reward']
                contexts.append((key, values))

                if len(contexts) % self.batchsize == 0:
                    for k, v in contexts:
                        self.learner.learn(k, 
                                           v['context'],
                                           v['action'],
                                           sumreward / self.batchsize)
                    sumreward = 0
                    contexts = []

            self.mem = {}
