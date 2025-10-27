
import torch
import torch.nn as nn
from src.models.model_wrapper import Llama3BModelWrapper

class ResidualEditor:
    def __init__(self, model_wrapper: Llama3BModelWrapper, layer_idx: int):
        self.model_wrapper = model_wrapper
        self.layer_idx = layer_idx
        self.delta = None
        self.handle = None
        self.module = self.model_wrapper.get_block_module(layer_idx)

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            others = output[1:]
        else:
            hidden = output
            others = None
        if self.delta is not None:
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + self.delta.to(hidden.device)
        if others is None:
            return hidden
        return (hidden,) + others

    def enable(self):
        if self.handle is None:
            self.handle = self.module.register_forward_hook(self._hook)

    def disable(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


