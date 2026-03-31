from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

from .config import MODEL_ID, DEVICE, BNB_CONFIG, GLOBAL_DEVICE

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_name_or_path: str = MODEL_ID, device: str = GLOBAL_DEVICE):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        logger.info(f"正在加载模型 {self.model_name_or_path} 到 {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True
        )
        # 推理加速
        try:
            self.model = torch.compile(self.model)
            logger.info("模型编译加速完成")
        except Exception as e:
            logger.warning(f"模型编译失败，使用原模型: {e}")
        logger.info("模型加载完成！")

    def encode_text(self, text: str) -> dict:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded.")
        return self.tokenizer(text, return_tensors="pt")

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been loaded.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)