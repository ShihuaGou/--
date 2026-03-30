from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import MODEL_ID, DEVICE


class ModelLoader:
    def __init__(self, model_name_or_path: str = MODEL_ID, device: str = DEVICE):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    def encode_text(self, text: str) -> dict:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded.")
        return self.tokenizer(text, return_tensors="pt")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been loaded.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
