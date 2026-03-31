import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import OrderedDict
from threading import Lock

import faiss
import numpy as np
import lz4.frame
import logging

from .config import (
    MEMORY_DB_PATH,
    MEMORY_DIM,
    ENABLE_COMPRESSION,
    VECTOR_QUANTIZE_TYPE,
    SHORT_MEMORY_LIMIT,
    HOT_MEMORY_LIMIT,
    SHORT_MEM_MAX_SIZE,
    HOT_MEM_MAX_SIZE,
    MAX_SUPER_SHORT,
)

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    memory_id: str
    content: str
    vector: np.ndarray
    memory_type: str = "short_term"
    task_type: Optional[str] = None
    agent_role: str = "global"
    access_count: int = 0
    create_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    is_compressed: bool = False
    importance_score: float = 0.0
    compressed_content: Optional[bytes] = None

@dataclass
class KnowledgeTriple:
    head: str
    relation: str
    tail: str

class UnifiedLowLevelSemanticMemory:
    def __init__(self):
        self.vector_dim = MEMORY_DIM
        self.enable_compression = ENABLE_COMPRESSION
        self.quantize_type = VECTOR_QUANTIZE_TYPE
        
        # 向量编码模型，CPU运行不占显存
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
            logger.info("向量编码模型加载完成")
        except ImportError:
            logger.warning("sentence-transformers未安装，使用随机向量")
            self.encoder = None
        
        # 结构化知识图谱
        self.kg_graph: List[KnowledgeTriple] = []
        self.protected_memory_ids: set = set()
        
        # 四级分层记忆
        self.super_short_mem = OrderedDict()
        self.max_super_short = MAX_SUPER_SHORT
        
        self.short_term_mem = OrderedDict()
        self.max_short_mem = SHORT_MEMORY_LIMIT
        self.short_mem_max_size = SHORT_MEM_MAX_SIZE
        self.short_mem_lock = Lock()
        
        self.hot_long_term_mem = OrderedDict()
        self.max_hot_mem = HOT_MEMORY_LIMIT
        self.hot_mem_max_size = HOT_MEM_MAX_SIZE
        self.hot_mem_lock = Lock()
        
        # GPU-FAISS索引
        self.cold_long_term_mem: List[MemoryItem] = []
        self._init_gpu_faiss_index()
        
        # 统计指标
        self.total_memory_usage = 0
        self.memory_fragmentation = 0.0
        logger.info("双模态统一记忆库初始化完成")

    def _init_gpu_faiss_index(self):
        """GPU-FAISS索引初始化"""
        try:
            if self.quantize_type == "INT8":
                cpu_index = faiss.IndexScalarQuantizer(self.vector_dim, faiss.ScalarQuantizer.QT_8bit)
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
            else:
                cpu_index = faiss.IndexFlatL2(self.vector_dim)
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
            self.id_to_memory: Dict[int, MemoryItem] = {}
            self.next_id = 0
            logger.info("GPU-FAISS索引初始化完成")
        except Exception as e:
            logger.warning(f"GPU-FAISS初始化失败，使用CPU索引: {e}")
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.id_to_memory = {}
            self.next_id = 0

    def _encode_text(self, text: str) -> np.ndarray:
        """文本转向量"""
        if self.encoder:
            vec = self.encoder.encode([text], normalize_embeddings=True)[0].astype(np.float32)
        else:
            vec = np.random.randn(self.vector_dim).astype(np.float32)
        return vec

    def _compress_content(self, content: str) -> bytes:
        """LZ4压缩"""
        if not self.enable_compression:
            return content.encode("utf-8")
        return lz4.frame.compress(content.encode("utf-8"))

    def _decompress_content(self, compressed_data: bytes) -> str:
        """LZ4解压缩"""
        if not self.enable_compression:
            return compressed_data.decode("utf-8")
        return lz4.frame.decompress(compressed_data).decode("utf-8")

    def _calc_memory_size(self, item: MemoryItem) -> int:
        """计算记忆占用"""
        size = 0
        if item.is_compressed:
            size += len(item.compressed_content)
        else:
            size += len(item.content.encode("utf-8"))
        size += item.vector.nbytes
        return size

    def add_knowledge_triple(self, triple: KnowledgeTriple):
        """添加知识三元组"""
        self.kg_graph.append(triple)

    def add_memory(
        self,
        content: str,
        memory_type: str = "short_term",
        task_type: Optional[str] = None,
        agent_role: str = "global",
        is_experience: bool = False,
        knowledge_triples: Optional[List[KnowledgeTriple]] = None
    ) -> str:
        """写入记忆"""
        vec = self._encode_text(content)
        memory_id = f"mem_{int(time.time() * 1000)}_{self.next_id}"
        current_time = time.time()

        is_compressed = self.enable_compression and memory_type != "short_term"
        compressed_content = self._compress_content(content) if is_compressed else None

        importance_score = 0.7 if is_experience else 0.3
        if knowledge_triples:
            importance_score += 0.2
            for triple in knowledge_triples:
                self.add_knowledge_triple(triple)

        item = MemoryItem(
            memory_id=memory_id,
            content=content if not is_compressed else "",
            compressed_content=compressed_content,
            vector=vec,
            memory_type=memory_type,
            task_type=task_type,
            agent_role=agent_role,
            access_count=0,
            create_time=current_time,
            last_access_time=current_time,
            is_compressed=is_compressed,
            importance_score=importance_score
        )

        item_size = self._calc_memory_size(item)

        if importance_score >= 0.8:
            self.protected_memory_ids.add(memory_id)

        if memory_type == "short_term":
            with self.short_mem_lock:
                self.short_term_mem[memory_id] = item
                if len(self.short_term_mem) > self.max_short_mem:
                    _, evicted = self.short_term_mem.popitem(last=False)
                    self.total_memory_usage -= self._calc_memory_size(evicted)
                self.total_memory_usage += item_size
        elif is_experience or memory_type == "long_term":
            with self.hot_mem_lock:
                self.hot_long_term_mem[memory_id] = item
                self.total_memory_usage += item_size
                while len(self.hot_long_term_mem) > self.max_hot_mem:
                    _, cold_item = self.hot_long_term_mem.popitem(last=False)
                    self._write_to_cold_index(cold_item)
                    self.total_memory_usage -= self._calc_memory_size(cold_item)
        else:
            self.super_short_mem[memory_id] = item
            if len(self.super_short_mem) > self.max_super_short:
                self.super_short_mem.popitem(last=False)
            self.total_memory_usage += item_size

        logger.debug(f"记忆写入成功，ID：{memory_id}")
        return memory_id

    def _write_to_cold_index(self, item: MemoryItem):
        """写入冷索引"""
        self.index.add(np.expand_dims(item.vector, axis=0))
        self.id_to_memory[self.next_id] = item
        self.next_id += 1
        self.cold_long_term_mem.append(item)

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        task_type: Optional[str] = None,
        agent_capability: Optional[List[str]] = None
    ) -> Tuple[List[MemoryItem], List[KnowledgeTriple]]:
        """检索记忆"""
        query_vec = self._encode_text(query)
        results: List[MemoryItem] = []

        # 检索热记忆
        for item in list(self.short_term_mem.values()) + list(self.hot_long_term_mem.values()):
            if memory_type and item.memory_type != memory_type:
                continue
            if task_type and item.task_type != task_type:
                continue
            results.append(item)

        # 检索冷记忆
        if len(results) < top_k and hasattr(self, 'index'):
            query_vec_reshaped = query_vec.reshape(1, -1).astype(np.float32)
            D, I = self.index.search(query_vec_reshaped, top_k - len(results))
            for idx in I[0]:
                if idx in self.id_to_memory:
                    item = self.id_to_memory[idx]
                    if item.is_compressed and not item.content:
                        item.content = self._decompress_content(item.compressed_content)
                    if item.memory_id not in self.hot_long_term_mem:
                        self.hot_long_term_mem[item.memory_id] = item
                    results.append(item)

        # 相似度排序
        results.sort(key=lambda x: np.dot(x.vector, query_vec), reverse=True)
        top_results = results[:top_k]

        # 结构化知识召回
        related_triples = []
        for triple in self.kg_graph:
            if query in triple.head or query in triple.tail:
                related_triples.append(triple)

        # 更新访问统计
        for item in top_results:
            item.access_count += 1
            item.last_access_time = time.time()
            item.importance_score = min(1.0, item.importance_score + 0.05 * item.access_count)
            if item.importance_score >= 0.8:
                self.protected_memory_ids.add(item.memory_id)

        return top_results, related_triples

    def distill_structured_experience(self, reasoning_trace: str, task_type: str, agent_id: str, is_strong_agent: bool):
        """经验蒸馏"""
        if not is_strong_agent:
            return
        reasoning_steps = [step.strip() for step in reasoning_trace.split("\n") if step.strip()]
        knowledge_triples = []
        for step in reasoning_steps:
            if "因为" in step and "所以" in step:
                head = step.split("因为")[-1].split("所以")[0].strip()
                tail = step.split("所以")[-1].strip()
                triple = KnowledgeTriple(head=head, relation="逻辑推导", tail=tail)
                knowledge_triples.append(triple)
        
        experience_content = f"""
        【任务类型】{task_type}
        【推理逻辑三元组】{[str(t.__dict__) for t in knowledge_triples]}
        【完整推理轨迹】{reasoning_trace}
        """
        self.add_memory(
            content=experience_content,
            memory_type="long_term",
            task_type=task_type,
            agent_role=agent_id,
            is_experience=True,
            knowledge_triples=knowledge_triples
        )
        logger.info(f"强智能体经验蒸馏完成")

    def get_memory_stats(self) -> Dict:
        """获取统计"""
        try:
            import psutil
            process = psutil.Process()
            rss_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            rss_mb = 0
        
        return {
            "total_memory_usage_mb": self.total_memory_usage / 1024 / 1024,
            "process_rss_mb": rss_mb,
            "short_mem_count": len(self.short_term_mem),
            "hot_mem_count": len(self.hot_long_term_mem),
            "cold_mem_count": len(self.cold_long_term_mem),
            "kg_triple_count": len(self.kg_graph),
            "protected_memory_count": len(self.protected_memory_ids)
        }