
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from src.probes.linear_probe import LinearProbe
from src.models.model_wrapper import Llama3BModelWrapper

class ProbeManager:
    def __init__(self, model_wrapper: Llama3BModelWrapper, layer_idx: int, pool_n: int, device: str, hidden_size: int):
        self.model_wrapper = model_wrapper
        self.layer_idx = layer_idx
        self.pool_n = pool_n
        self.device = device
        self.hidden_size = hidden_size

        self.conf_probe = LinearProbe(hidden_size).to(self.device)
        self.truth_probe = LinearProbe(hidden_size).to(self.device)
        self.harm_probe = LinearProbe(hidden_size).to(self.device)
        self.benign_probe = LinearProbe(hidden_size).to(self.device)
        self.proc_probe = LinearProbe(hidden_size).to(self.device)

    @staticmethod
    def norm_entropy(probs: torch.Tensor) -> torch.Tensor:
        H = -(probs * (probs + 1e-9).log()).sum(dim=-1)  # [B]
        H_max = math.log(probs.size(-1))
        return H / (H_max + 1e-9)

    @staticmethod
    def temporal_kl(p_t: torch.Tensor, p_tm1: torch.Tensor) -> torch.Tensor:
        return (p_t * ((p_t + 1e-9).log() - (p_tm1 + 1e-9).log())).sum(dim=-1)

    @torch.no_grad()
    def collect_confidence_dataset(self, prompts: list[str], alpha: float = 1.0, beta: float = 0.0, gamma: float = 1.0):
        X, y = [], []
        for text in prompts:
            enc = self.model_wrapper.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            logits, hids = self.model_wrapper.forward_logits_and_hiddens(enc["input_ids"])
            H = hids[self.layer_idx]  # [1, T, H]
            P = torch.softmax(logits[0], dim=-1)  # [T, V]
            T = P.size(0)

            # temporal stability sˆ(KL)
            kl_scale = 5.0
            s_kl = []
            s_kl.append(torch.tensor([1.0], device=self.device))
            for t in range(1, T):
                kl = self.temporal_kl(P[t].unsqueeze(0), P[t-1].unsqueeze(0))
                s = 1.0 - torch.clamp(kl / kl_scale, 0.0, 1.0)
                s_kl.append(s)
            s_kl = torch.stack(s_kl).squeeze(-1)  # [T]

            cH = 1.0 - self.norm_entropy(P)  # [T]
            cmax = P.max(dim=-1).values  # [T]
            target = alpha * cH + beta * cmax + gamma * s_kl
            target = torch.clamp(target, 0.0, 1.0)

            LN = nn.LayerNorm(H.size(-1)).to(self.device)
            for t in range(T):
                s = max(0, t - self.pool_n + 1)
                pooled = LN(H[0, s:t+1, :].mean(dim=0, keepdim=True))
                X.append(pooled.squeeze(0).detach().cpu().numpy())
                y.append(float(target[t].item()))
        return np.stack(X, axis=0), np.array(y)

    def train_conf_probe(self, X: np.ndarray, y: np.ndarray, epochs: int = 8, batch_size: int = 256, lr: float = 1e-3, l2: float = 1e-4):
        Xtr, Xdv, ytr, ydv = train_test_split(X, y, test_size=0.1, random_state=42)
        tr = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
        dv = TensorDataset(torch.tensor(Xdv, dtype=torch.float32), torch.tensor(ydv, dtype=torch.float32))
        tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
        dv_loader = DataLoader(dv, batch_size=batch_size, shuffle=False)

        self.conf_probe.train()
        opt = optim.Adam(self.conf_probe.parameters(), lr=lr, weight_decay=l2)
        lossf = nn.MSELoss()

        for ep in range(epochs):
            tot = 0.0
            for xb, yb in tr_loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                pred, _ = self.conf_probe(xb)
                loss = lossf(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * xb.size(0)
            print(f"[conf] epoch {ep+1}/{epochs} mse={tot/len(tr):.4f}")

        # Platt calibration on dev
        self.conf_probe.eval()
        with torch.no_grad():
            Z, Y = [], []
            for xb, yb in dv_loader:
                xb = xb.to(self.device)
                _, logit = self.conf_probe(xb)
                Z.append(logit.cpu().numpy())
                Y.append(yb.numpy())
            Z = np.concatenate(Z).reshape(-1)
            Y = np.concatenate(Y).reshape(-1)

            A, B = 1.0, 0.0
            for _ in range(1000):
                pred = 1.0 / (1.0 + np.exp(-(A * Z + B)))
                gradA = np.mean((pred - Y) * Z)
                gradB = np.mean(pred - Y)
                A -= 1e-3 * gradA
                B -= 1e-3 * gradB
            self.conf_probe.calib_A = float(A); self.conf_probe.calib_B = float(B)
            print(f"[conf] Platt calib: A={A:.3f}, B={B:.3f}")
        return self.conf_probe

    def train_binary_probe(self, probe: LinearProbe, X: np.ndarray, y: np.ndarray, epochs: int = 6, batch_size: int = 256, lr: float = 1e-3, l2: float = 1e-4):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        probe.train()
        opt = optim.Adam(probe.parameters(), lr=lr, weight_decay=l2)
        lossf = nn.BCELoss()
        for ep in range(epochs):
            tot = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device); yb = yb.to(self.device)
                pred, _ = probe(xb)
                loss = lossf(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * xb.size(0)
            print(f"[{probe.__class__.__name__}] epoch {ep+1} bce={tot/len(ds):.4f}")
        probe.eval()
        return probe

    @torch.no_grad()
    def collect_features_for_prompts(self, prompts: list[str]):
        X = []
        LN = nn.LayerNorm(self.hidden_size).to(self.device) # Initialize LayerNorm once
        for text in prompts:
            enc = self.model_wrapper.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            logits, hids = self.model_wrapper.forward_logits_and_hiddens(enc["input_ids"])
            H = hids[self.layer_idx]  # [1, T, H]
            
            # Pool across the sequence dimension to get one vector per prompt
            # Similar to how pooled is calculated in collect_confidence_dataset, but for the whole prompt
            pooled_h = LN(H[0, :, :].mean(dim=0, keepdim=True))
            X.append(pooled_h.squeeze(0).detach().cpu().numpy())
        return np.stack(X, axis=0)


