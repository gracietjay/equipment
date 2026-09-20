"""
快路径增强检索节点
缓存命中后用 compressed_history 增强 query，单路 Milvus 检索，跳过 HyDE/Web/KG/RRF
"""
import sys
import os
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

FAST_PATH_TOPK = 8
FAST_PATH_LIMIT = 5


def node_enhanced_search(state):
    """
    快路径检索节点
    用 compressed_history 增强 query 语义，item_names 做 Milvus 过滤，单路混合检索
    输出 embedding_chunks 和 rrf_chunks（跳过 RRF，直接喂给 rerank）
    """
    logger.info("--- node_enhanced_search 开始处理 ---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    query = state.get("rewritten_query") or state.get("original_query", "")
    compressed_history = state.get("compressed_history", "")
    item_names = state.get("item_names") or []

    # 用 compressed_history 增强 query
    if compressed_history:
        enhanced_query = f"之前讨论的背景：{compressed_history}\n当前新问题：{query}"
    else:
        enhanced_query = query

    # 生成 Embedding
    embeddings = generate_embeddings([enhanced_query])
    dense_vec = embeddings.get("dense")[0]
    sparse_vec = embeddings.get("sparse")[0]

    # 构造 Milvus 过滤表达式
    collection_name = os.environ.get("CHUNKS_COLLECTION")
    if not collection_name:
        logger.error("CHUNKS_COLLECTION 未配置")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"embedding_chunks": [], "rrf_chunks": []}

    if not item_names:
        expr = None
    else:
        quoted = ", ".join(f'"{v}"' for v in item_names)
        expr = f"item_name in [{quoted}]"

    # 混合检索
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec, sparse_vector=sparse_vec,
        expr=expr, limit=FAST_PATH_TOPK
    )
    client = get_milvus_client()
    res = hybrid_search(
        client=client, collection_name=collection_name, reqs=reqs,
        ranker_weights=(0.8, 0.2), norm_score=True, limit=FAST_PATH_LIMIT,
        output_fields=["chunk_id", "content", "item_name", "title"]
    )

    hit_count = len(res[0]) if res and len(res) > 0 else 0

    # 兜底：filter 太严返回 0 结果时，去掉 filter 重试
    if hit_count == 0 and expr is not None:
        logger.warning(f"带 filter 检索返回 0 结果，去掉 filter 兜底重试")
        reqs_fallback = create_hybrid_search_requests(
            dense_vector=dense_vec, sparse_vector=sparse_vec,
            expr=None, limit=FAST_PATH_TOPK
        )
        res = hybrid_search(
            client=client, collection_name=collection_name, reqs=reqs_fallback,
            ranker_weights=(0.8, 0.2), norm_score=True, limit=FAST_PATH_LIMIT,
            output_fields=["chunk_id", "content", "item_name", "title"]
        )
        hit_count = len(res[0]) if res and len(res) > 0 else 0

    chunks = res[0] if res else []
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info(f"--- node_enhanced_search 处理结束，命中 {hit_count} 条 ---")

    return {"embedding_chunks": chunks, "rrf_chunks": chunks}
