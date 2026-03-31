from __future__ import annotations

import numpy as np
import torch
import logging
from typing import Optional, List, Dict

from .config import MEMORY_DIM, LOGS_DIR
from .memory import MemoryItem, UnifiedLowLevelSemanticMemory, KnowledgeTriple
from .model_loader import ModelLoader

logger = logging.getLogger(__name__)

class ULMMarAgent:
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.memory_store = UnifiedLowLevelSemanticMemory()
        logger.info("ULM-MAR代理初始化完成")

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
        memory_summary = str(self.memory_store.get_memory_stats())
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
            memory_id=str(len(self.memory_store.short_term_mem) + len(self.memory_store.hot_long_term_mem) + 1),
            content=text,
            vector=embeddings,
            memory_type="short_term",
        )
        self.memory_store.add_memory(item.content, memory_type=item.memory_type)

    def _encode_to_vector(self, text: str) -> np.ndarray:
        return self.memory_store._encode_text(text)
