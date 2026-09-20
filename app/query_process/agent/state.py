from typing_extensions import TypedDict
from typing import List, Optional, Any


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    v3.0 新增：评估字段和中断字段（Ragas 评估框架集成）
    """
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    kg_chunks: list  # 图谱检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: list  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记

    # ==================== V3.0 新增：缓存/来源标记 ====================
    from_cache: Optional[bool]  # 答案是否来自 Redis 缓存命中（True=缓存，None/False=正常检索）

    # 记忆分层字段
    cache_hit: Optional[bool]  # Redis 短期记忆命中标记
    compressed_history: Optional[str]  # 命中摘要的压缩历史，用于增强 query
    history_item_names: Optional[list]  # MongoDB 长期记忆中提取的实体名

    # ==================== V3.0 新增：Ragas 评估字段 ====================
    # --- 评估配置 ---
    eval_enabled: Optional[bool]  # 是否启用评估（默认 True，可关闭）
    eval_ground_truth: Optional[str]  # 评估时的标准答案上下文（从测试集读取）

    # --- 检索评估字段 ---
    eval_retrieval_precision: Optional[float]  # ContextPrecision 得分
    eval_retrieval_recall: Optional[float]  # ContextRecall 得分
    eval_retrieval_passed: Optional[bool]  # 检索评估是否通过
    eval_retrieval_fail_count: Optional[int]  # 检索评估连续失败次数

    # --- 生成评估字段 ---
    eval_generation_relevancy: Optional[float]  # ResponseRelevancy 得分
    eval_generation_faithfulness: Optional[float]  # Faithfulness 得分
    eval_generation_passed: Optional[bool]  # 生成评估是否通过

    # --- 中断字段 ---
    needs_interrupt: Optional[bool]  # 是否需要触发中断（True=需等待人工审核）
    interrupt_stage: Optional[str]  # 中断发生的阶段（retrieval / generation）
    interrupt_id: Optional[str]  # 中断唯一标识
    human_reviewed: Optional[bool]  # 是否已通过人工审核
    human_review_action: Optional[str]  # 人工审核动作（approved / rejected / modified）
