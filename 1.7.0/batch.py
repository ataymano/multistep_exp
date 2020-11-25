from typing import Hashable, Sequence, Dict, Any

class Batched:
    def __init__(self, delay: int, batchsize: int, learner):
        self.learner = learner()
        self.batchsize = batchsize
        self.delay = delay
        self.mem = {}

        assert self.delay % self.batchsize == 0

    def init(self):
        self.learner.init()

    @property
    def family(self) -> str:
        return "Batched Learner"

    @property
    def params(self) -> Dict[str,Any]:
        return { 
                 #**self.learner.params(),
                 **{ 'delay': self.delay, 'batchsize': self.batchsize },
               }

    def choose(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> int:
        """Choose which action index to take."""
        return self.learner.choose(key, context, actions)

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float) -> None:
        """Learn about the result of an action that was taken in a context."""

        self.mem[key] = { 'context': context,
                          'action': action,
                          'reward': reward
                        }

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