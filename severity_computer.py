
import torch

class SeverityComputer:
    def __init__(self, tau_h: float = 0.5, tau_conf_hi: float = 0.8, tau_conf_mid: float = 0.5,
                 gamma_h: float = 0.15, gamma_c: float = 0.15, gamma_q: float = 0.15):
        self.tau_h = tau_h
        self.tau_conf_hi = tau_conf_hi
        self.tau_conf_mid = tau_conf_mid
        self.gamma_h = gamma_h
        self.gamma_c = gamma_c
        self.gamma_q = gamma_q

    @staticmethod
    def sigmoid_sched(x: torch.Tensor, tau: float, gamma: float) -> torch.Tensor:
        return torch.sigmoid((x - tau) / (gamma + 1e-9))

    def contextual_harm(self, h: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return h * (1.0 - b) * p

    def s_harm(self, h_ctx: float, c: float, device: str) -> float:
        a = self.sigmoid_sched(torch.tensor([h_ctx], device=device), self.tau_h, self.gamma_h)
        b = self.sigmoid_sched(torch.tensor([c], device=device), self.tau_conf_hi, self.gamma_c)
        return (a * b).item()

    def s_truth(self, q: float, c: float, h_ctx: float, device: str) -> float:
        a = self.sigmoid_sched(torch.tensor([q], device=device), 0.8, self.gamma_q)
        b = self.sigmoid_sched(torch.tensor([self.tau_conf_mid - c], device=device), 0.0, self.gamma_c)
        hc = self.sigmoid_sched(torch.tensor([h_ctx], device=device), self.tau_h, self.gamma_h)
        return (a * b * (1.0 - hc)).item()


