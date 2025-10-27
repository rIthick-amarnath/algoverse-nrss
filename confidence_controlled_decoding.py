
import math
import torch
import torch.nn as nn
import numpy as np

from src.models.model_wrapper import Llama3BModelWrapper
from src.probes.probe_manager import ProbeManager
from src.models.residual_editor import ResidualEditor
from src.evaluation.severity_computer import SeverityComputer
from src.evaluation.temporal_guard import TemporalGuard

class ConfidenceControlledDecoding:
    def __init__(self, model_wrapper: Llama3BModelWrapper, probe_manager: ProbeManager, config: dict):
        self.model_wrapper = model_wrapper
        self.probe_manager = probe_manager
        self.config = config

        self.HARM_THD = config.get("HARM_THD", 0.50)
        self.HIGH_CONF_THD = config.get("HIGH_CONF_THD", 0.80)
        self.LOW_CONF_THD = config.get("LOW_CONF_THD", 0.30)
        self.TRUTH_THD = config.get("TRUTH_THD", 0.80)
        self.ETA_H = config.get("ETA_H", 2.0)
        self.ETA_Q = config.get("ETA_Q", 2.0)
        self.MU = config.get("MU", 0.2)
        self.EPSILON = config.get("EPSILON", 5.0)
        self.LAYER_IDX = config.get("LAYER_IDX", self.model_wrapper.model.config.num_hidden_layers - 2)
        self.POOL_N = config.get("POOL_N", 32)

        self.sev = SeverityComputer(
            tau_h=self.HARM_THD,
            tau_conf_hi=self.HIGH_CONF_THD,
            tau_conf_mid=0.50
        )
        self.guard = TemporalGuard(K=config.get("GUARD_K", 5), tau_low=config.get("GUARD_TAU_LOW", 0.20), doomed_k=config.get("GUARD_DOOMED_K", 5))
        self.editor = ResidualEditor(model_wrapper, self.LAYER_IDX)

    def pooled_hidden_from(self, hids: tuple) -> torch.Tensor:
        H = hids[self.LAYER_IDX]  # [1, T, H]
        LN = nn.LayerNorm(H.size(-1)).to(self.model_wrapper.device)
        s = max(0, H.size(1) - self.POOL_N)
        pooled = LN(H[:, s:, :].mean(dim=1))  # [1, H]
        return pooled

    @torch.no_grad()
    def decode_with_conf_control(self, prompt: str, max_new_tokens: int = 64, temperature: float = None, top_p: float = None):
        inputs = self.model_wrapper.tokenizer(prompt, return_tensors="pt").to(self.model_wrapper.device)
        generated = inputs["input_ids"][0].tolist()
        self.guard.buf, self.guard.low_run = [], 0

        for step in range(max_new_tokens):
            # Pass 1: read probes from unedited hidden at layer l
            out = self.model_wrapper.model(
                input_ids=torch.tensor([generated]).to(self.model_wrapper.device),
                output_hidden_states=True,
                use_cache=False
            )
            hids = out.hidden_states
            pooled = self.pooled_hidden_from(hids)  # [1, H]

            # Probes (calibrated if available)
            c_hat = self.probe_manager.conf_probe.predict_calibrated(pooled).item()
            q_hat = self.probe_manager.truth_probe.predict_calibrated(pooled).item()
            h_hat = self.probe_manager.harm_probe.predict_calibrated(pooled).item()
            b_hat = self.probe_manager.benign_probe.predict_calibrated(pooled).item()
            p_hat = self.probe_manager.proc_probe.predict_calibrated(pooled).item()

            # Context-aware harm
            h_ctx = self.sev.contextual_harm(
                torch.tensor([h_hat]), torch.tensor([b_hat]), torch.tensor([p_hat])
            ).item()

            # Temporal guard
            self.guard.push(c_hat)
            if self.guard.doomed():
                print("[controller] Early stop: prolonged low confidence.")
                break

            # Compute target confidence by smooth severities
            target_c = None
            if (h_ctx >= self.HARM_THD) and (c_hat >= self.HIGH_CONF_THD):
                s_h = self.sev.s_harm(h_ctx, c_hat, self.model_wrapper.device)
                target_c = max(0.0, min(1.0, c_hat * math.exp(-self.ETA_H * s_h)))
            elif (h_ctx < self.HARM_THD) and (q_hat >= self.TRUTH_THD) and (c_hat <= self.LOW_CONF_THD):
                s_q = self.sev.s_truth(q_hat, c_hat, h_ctx, self.model_wrapper.device)
                target_c = max(0.0, min(1.0, 1.0 - (1.0 - c_hat) * math.exp(-self.ETA_Q * s_q)))

            # Compute delta along w_c (minimal-norm to reach target_c)
            delta_vec = None
            if target_c is not None and (1e-6 < target_c < 1.0 - 1e-6):
                Wc = self.probe_manager.conf_probe.linear.weight.data.squeeze(0)  # [H]
                bc = self.probe_manager.conf_probe.linear.bias.data.item()
                current_logit = (Wc @ pooled.squeeze(0)).item() + bc
                target_logit = math.log(target_c / (1.0 - target_c + 1e-12) + 1e-12)
                delta_logit = target_logit - current_logit
                wc_norm2 = (Wc.norm()**2 + 1e-12).item()
                delta_vec = (delta_logit / wc_norm2) * Wc
                # clip and scale
                dnorm = delta_vec.norm().item()
                if dnorm > self.EPSILON:
                    delta_vec = delta_vec * (self.EPSILON / (dnorm + 1e-12))
                delta_vec = self.MU * delta_vec

            # Pass 2: run with hook applying delta at layer l
            if delta_vec is not None:
                self.editor.delta = delta_vec
                self.editor.enable()
            else:
                self.editor.delta = None
                self.editor.enable()  # enable with None is harmless

            out2 = self.model_wrapper.model(
                input_ids=torch.tensor([generated]).to(self.model_wrapper.device),
                use_cache=False
            )
            self.editor.disable()
            logits = out2.logits[:, -1, :]

            # Sampling
            if (temperature is None) and (top_p is None):
                next_id = int(torch.argmax(logits, dim=-1))
            else:
                probs = torch.softmax(logits / (temperature or 1.0), dim=-1).squeeze(0)
                if top_p is not None:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-12)
                    next_id = int(sorted_idx[torch.multinomial(sorted_probs, 1)])
                else:
                    next_id = int(torch.multinomial(probs, 1))

            generated.append(next_id)
            if next_id == self.model_wrapper.tokenizer.eos_token_id:
                break

        return self.model_wrapper.tokenizer.decode(generated[inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def eval_controller(self, prompts: list[str], gold_answers: list[str] = None, max_new_tokens: int = 64):
        results = []
        for i, p in enumerate(prompts):
            text = self.decode_with_conf_control(p, max_new_tokens=max_new_tokens)

            enc = self.model_wrapper.tokenizer(p + text, return_tensors="pt").to(self.model_wrapper.device)
            out = self.model_wrapper.model(**enc, output_hidden_states=True, use_cache=False)
            pooled = self.pooled_hidden_from(out.hidden_states)

            c_final = self.probe_manager.conf_probe.predict_calibrated(pooled).item()
            q_final = self.probe_manager.truth_probe.predict_calibrated(pooled).item()
            h_final = self.probe_manager.harm_probe.predict_calibrated(pooled).item()
            b_final = self.probe_manager.benign_probe.predict_calibrated(pooled).item()
            p_final = self.probe_manager.proc_probe.predict_calibrated(pooled).item()

            correct = None
            if gold_answers is not None and gold_answers[i] is not None:
                correct = (text.strip().lower() == gold_answers[i].strip().lower())

            h_ctx = self.sev.contextual_harm(
                torch.tensor([h_final]), torch.tensor([b_final]), torch.tensor([p_final])
            ).item()
            harmful = (h_ctx >= self.HARM_THD)

            results.append(dict(output=text, final_conf=c_final, correct=correct, harmful=harmful, q=q_final, h=h_final, b=b_final, p=p_final))

        # Placeholder for ECE calculation - needs proper implementation
        def expected_calibration_error(confidences, accuracies, num_bins=10):
            if not confidences or not accuracies:
                return float('nan')

            bin_edges = np.linspace(0, 1, num_bins + 1)
            ece = 0.0
            total_samples = len(confidences)

            for i in range(num_bins):
                lower_bound = bin_edges[i]
                upper_bound = bin_edges[i+1]

                bin_indices = [j for j, conf in enumerate(confidences) if lower_bound <= conf < upper_bound]
                if not bin_indices:
                    continue

                bin_confidences = [confidences[j] for j in bin_indices]
                bin_accuracies = [accuracies[j] for j in bin_indices]

                avg_bin_confidence = sum(bin_confidences) / len(bin_confidences)
                avg_bin_accuracy = sum(bin_accuracies) / len(bin_accuracies)

                ece += (len(bin_indices) / total_samples) * abs(avg_bin_confidence - avg_bin_accuracy)
            return ece

        acc_list, conf_list = [], []
        for r in results:
            if r["correct"] is not None:
                acc_list.append(1.0 if r["correct"] else 0.0)
                conf_list.append(r["final_conf"])

        acc = np.mean(acc_list) if acc_list else float("nan")
        ece = expected_calibration_error(conf_list, acc_list) if acc_list else float("nan")
        harm_rate = np.mean([r["harmful"] for r in results]) if results else float("nan")

        print(f"[EVAL] Accuracy: {acc:.3f} | Harm rate: {harm_rate:.3f} | ECE: {ece:.3f}")
        return results


