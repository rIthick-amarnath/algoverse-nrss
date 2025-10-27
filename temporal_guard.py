
import numpy as np

class TemporalGuard:
    def __init__(self, K: int = 5, tau_low: float = 0.20, doomed_k: int = 5):
        self.K = K
        self.buf = []
        self.tau_low = tau_low
        self.doomed_k = doomed_k
        self.low_run = 0

    def push(self, c: float):
        self.buf.append(float(c))
        if len(self.buf) > self.K:
            self.buf.pop(0)
        if c < self.tau_low:
            self.low_run += 1
        else:
            self.low_run = 0

    def stability(self) -> float:
        if len(self.buf) < 2:
            return 1.0
        return 1.0 - float(np.var(np.array(self.buf)))

    def doomed(self) -> bool:
        return self.low_run >= self.doomed_k


