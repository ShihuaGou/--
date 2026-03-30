import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from .config import MEMORY_DB_PATH, MEMORY_DIM


@dataclass
class MemoryItem:
    memory_id: str
    content: str
    vector: np.ndarray
    memory_type: str = "short_term"
    create_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    importance_score: float = 0.0


class MemoryStore:
    def __init__(self, dimension: int = MEMORY_DIM):
        self.dimension = dimension
        self.items: List[MemoryItem] = []
        self.index = faiss.IndexFlatIP(dimension)
        MEMORY_DB_PATH.mkdir(parents=True, exist_ok=True)

    def add_memory(self, item: MemoryItem) -> None:
        if item.vector.shape != (self.dimension,):
            raise ValueError(f"Memory vector must be shape ({self.dimension},), got {item.vector.shape}")
        self.items.append(item)
        self.index.add(np.expand_dims(item.vector, axis=0))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[MemoryItem]:
        if len(self.items) == 0:
            return []
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        result = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.items):
                continue
            result.append(self.items[idx])
        return result

    def rebuild_index(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        if self.items:
            vectors = np.vstack([item.vector for item in self.items]).astype(np.float32)
            self.index.add(vectors)

    def summary(self) -> str:
        total = len(self.items)
        return f"Memory store contains {total} items."
