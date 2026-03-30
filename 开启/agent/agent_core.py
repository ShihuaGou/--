from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from .config import MEMORY_DIM
from .memory import MemoryItem, MemoryStore
from .model_loader import ModelLoader


class ULMMarAgent:
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.memory_store = MemoryStore(dimension=MEMORY_DIM)

    def build_prompt(self, instruction: str, context: Optional[str] = None, memory_summary: Optional[str] = None) -> str:
        prompt_parts = ["You are a multimodal reasoning assistant."]
        if context:
            prompt_parts.append(f"Context: {context}")
        if memory_summary:
            prompt_parts.append(f"Memory summary: {memory_summary}")
        prompt_parts.append(f"Instruction: {instruction}")
        prompt_parts.append("Please provide a clear, concise answer with reasoning.")
        return "\n".join(prompt_parts)

    def process_query(self, query_text: str, context: Optional[str] = None) -> tuple[str, str]:
        memory_summary = self.memory_store.summary()
        prompt = self.build_prompt(query_text, context=context, memory_summary=memory_summary)
        output = self.generate_response(prompt)
        self.add_memory(query_text, output)
        return output, memory_summary

    def generate_response(self, prompt: str, max_tokens: int = 256) -> str:
        return self.model_loader.generate(prompt, max_new_tokens=max_tokens)

    def add_memory(self, input_text: str, output_text: str) -> None:
        if self.model_loader.tokenizer is None:
            return
        text = f"USER: {input_text}\nAGENT: {output_text}"
        embeddings = self._encode_to_vector(text)
        item = MemoryItem(
            memory_id=str(len(self.memory_store.items) + 1),
            content=text,
            vector=embeddings,
            memory_type="short_term",
        )
        self.memory_store.add_memory(item)

    def _encode_to_vector(self, text: str) -> np.ndarray:
        tokenized = self.model_loader.encode_text(text)
        input_ids = tokenized["input_ids"].to(self.model_loader.model.device)
        with torch.no_grad():
            outputs = self.model_loader.model(input_ids=input_ids)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
        if pooled.shape[0] != MEMORY_DIM:
            pooled = np.resize(pooled, (MEMORY_DIM,))
        return pooled.astype(np.float32)
