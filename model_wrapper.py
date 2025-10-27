
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llama3BModelWrapper:
    def __init__(self, model_name: str = "meta-llama/Llama-3-8B", device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        self.model.eval()

    def get_block_module(self, layer_idx: int):
        """Return the layer block module to hook. Supports GPT-2 family and LLaMA/Mistral family."""
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        raise RuntimeError("Unsupported model architecture for hooking.")

    def forward_logits_and_hiddens(self, input_ids):
        out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        return out.logits, out.hidden_states


