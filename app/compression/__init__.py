"""
压缩模块 - 初始化
功能：汇聚 compression 子模块的核心接口，对外提供统一入口
"""
from app.compression.relevance_judger import (
    judge_relevance_cross_encoder,
    judge_relevance_embedding,
    judge_relevance_llm,
    judge_relevance,
    check_cache_relevance
)
from app.compression.cache_manager import (
    RedisCacheManager,
    get_cache_manager
)
from app.compression.context_compact_engine import (
    ContextCompactEngine,
    get_compact_engine
)
