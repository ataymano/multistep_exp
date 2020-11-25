from typing import Hashable, Sequence, Dict, Any

class Advantage:
    def __init__(self, seed: int, flags: str, learner):
        self.learner = learner()
        self.flags = flags
        self.seed = seed
        self.baseline=None

    def init(self):
        from os import devnull
        from coba import execution

        with open(devnull, 'w') as f, execution.redirect_stderr(f):
            from vowpalwabbit import pyvw
            self.baseline = pyvw.vw(f'--quiet ${self.flags} --random_seed {self.seed}')

    def tovw(self, context, reward, prob):
        assert type(context) is tuple, context

        return '\n'.join([
            f'{reward} {1.0/prob} | ' 
          + ' '.join([ f'{k+1}:{v}' for k, v in enumerate(context) if v != 0 ])
          ])

    @property
    def family(self) -> str:
        return "Advantage Wrapper"

    @property
    def params(self) -> Dict[str,Any]:
        return self.learner.params()

    def choose(self, key: int, context: Hashable, actions: Sequence[Hashable]) -> int:
        return self.learner.choose(key, context, actions)

    def learn(self, key: int, context: Hashable, action: Hashable, reward: float) -> None:
        prob = self.learner._probs[key]
        exstr = self.tovw(context, reward, prob)
        vhat = self.baseline.predict(exstr)
        self.baseline.learn(exstr)
        self.learner.learn(key, context, action, reward - vhat)