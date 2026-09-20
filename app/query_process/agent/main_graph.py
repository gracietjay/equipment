"""
查询流程 LangGraph 主图

v2.0: Redis 缓存检查
v3.0: Ragas 评估 + 中断机制
v4.0: 记忆分层 — 快路径（摘要导航）+ 增强路径（长期记忆兜底）
"""
from langgraph.graph import StateGraph, END
from app.query_process.agent.state import QueryGraphState
# 导入所有节点函数
from app.query_process.agent.nodes.node_cache_check import node_cache_check
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
# V3.0 新增：评估节点
from app.query_process.agent.nodes.node_eval_retrieval import node_eval_retrieval
from app.query_process.agent.nodes.node_eval_generation import node_eval_generation
# V4.0 新增：快路径增强检索节点
from app.query_process.agent.nodes.node_enhanced_search import node_enhanced_search

from app.core.logger import logger

# 初始化状态图
builder = StateGraph(QueryGraphState)

# ==================== V4.0 注册所有节点 ====================
# --- 基础流程节点 ---
builder.add_node("node_cache_check", node_cache_check)           # 【V2.0】Redis缓存检查
builder.add_node("node_item_name_confirm", node_item_name_confirm)  # 确认商品
builder.add_node("node_multi_search", lambda x: x)               # 虚拟节点：多路搜索分叉点
builder.add_node("node_search_embedding", node_search_embedding)  # 向量搜索
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_query_kg", node_query_kg)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_join", lambda x: {})                      # 虚拟节点：多路搜索合并点
builder.add_node("node_rrf", node_rrf)                           # RRF排序
builder.add_node("node_rerank", node_rerank)                     # 重排
# --- 【V3.0新增】评估节点 ---
builder.add_node("node_eval_retrieval", node_eval_retrieval)     # 检索评估（ContextPrecision+Recall）
builder.add_node("node_answer_output", node_answer_output)       # 答案生成
builder.add_node("node_eval_generation", node_eval_generation)   # 【V3.0新增】生成评估（ResponseRelevancy+Faithfulness）
# --- V4.0 快路径节点 ---
builder.add_node("node_enhanced_search", node_enhanced_search)   # 摘要导航 + 单路检索

# ==================== V3.0 设置起点 ====================
builder.set_entry_point("node_cache_check")


# ==================== 条件路由函数 ====================

def route_after_cache_check(state: QueryGraphState):
    """缓存检查后路由：有answer直接输出 / 命中走快路径 / 未命中走增强路径"""
    if state.get("answer"):
        return "node_answer_output"
    if state.get("cache_hit"):
        logger.info("[路由] 快路径: enhanced_search")
        return "node_enhanced_search"
    logger.info("[路由] 增强路径: item_name_confirm")
    return "node_item_name_confirm"


def route_after_item_confirm(state: QueryGraphState):
    """意图确认后的条件路由"""
    if state.get("answer"):
        return "node_answer_output"
    return "node_multi_search"


def route_after_rerank(state: QueryGraphState):
    """重排后路由：中断检查 → 答案生成"""
    if state.get("needs_interrupt") and state.get("interrupt_stage") == "retrieval":
        logger.warning(f"[路由] 检索评估中断，interrupt_id={state.get('interrupt_id')}")
        return END
    return "node_answer_output"


def route_after_answer_output(state: QueryGraphState):
    """答案生成后路由：生成评估中断检查"""
    if state.get("needs_interrupt") and state.get("interrupt_stage") == "generation":
        logger.warning(f"[路由] 生成评估中断，interrupt_id={state.get('interrupt_id')}")
    return END


# ==================== V3.0 边的注册 ====================

# 1. 缓存检查 → 快路径 / 增强路径 / 直接输出
builder.add_conditional_edges("node_cache_check", route_after_cache_check)

# 1.5. 快路径：增强检索 → 重排
builder.add_edge("node_enhanced_search", "node_rerank")

# 2. 意图确认 → (条件路由) → 多路搜索 / 答案输出
builder.add_conditional_edges("node_item_name_confirm", route_after_item_confirm)

# 3. 并发执行四路搜索
builder.add_edge("node_multi_search", "node_search_embedding")
builder.add_edge("node_multi_search", "node_search_embedding_hyde")
builder.add_edge("node_multi_search", "node_web_search_mcp")
builder.add_edge("node_multi_search", "node_query_kg")

# 4. 四路搜索 → 结果合并
builder.add_edge("node_search_embedding", "node_join")
builder.add_edge("node_search_embedding_hyde", "node_join")
builder.add_edge("node_web_search_mcp", "node_join")
builder.add_edge("node_query_kg", "node_join")

# 5. join → RRF → rerank → 检索评估
builder.add_edge("node_join", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_eval_retrieval")

# 6. 检索评估 → 答案生成 / 中断
builder.add_conditional_edges("node_eval_retrieval", route_after_rerank)

# 7. 答案生成 → 生成评估
builder.add_edge("node_answer_output", "node_eval_generation")

# 8. 生成评估 → 结束
builder.add_edge("node_eval_generation", END)

# 编译生成可执行的 Runnable 应用
query_app = builder.compile()
