"""
相关性判断模块 - 判断用户新提问与缓存 Q&A 摘要之间的语义相关性
功能：采用三级递进式判断策略，在准确性和效率之间取得平衡
      1级（快速）：BGE-M3 Embedding 余弦相似度（Bi-Encoder 快速预筛，O(n) 级别）
      2级（精确）：BGE-Reranker 交叉编码器评分（Cross-Encoder 精确排序）
      3级（兜底）：LLM 语义相关性判断（大模型深度语义理解，作为最后兜底）

方法出处与引用：
  - 交叉编码器（Cross-Encoder）范式：
    Nogueira & Cho, "Passage Re-ranking with BERT" (2019)
    https://arxiv.org/abs/1901.04085
  - BGE（BAAI General Embedding）系列模型：
    Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding" (SIGIR 2023)
    https://arxiv.org/abs/2309.07597
  - Bi-Encoder（双编码器）检索范式：
    Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (EMNLP 2019)
    https://arxiv.org/abs/1908.10084
  - LLM-as-Judge 范式：
    Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023)
    https://arxiv.org/abs/2306.05685

设计思路：
  - 优先使用最快速的 Embedding 相似度做预筛选（O(n) 级别，适合全量候选）
  - 候选量缩小后使用 Cross-Encoder 做精确评分（O(1) 高质量评分）
  - 当模型不可用时，降级到 LLM 判断（最慢但最灵活）
  - 三级递进策略参考了当前工业界「检索→精排→重排」的标准 pipeline
"""
import json
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv

from app.conf.redis_config import redis_config
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.reranker_utils import get_reranker_model
from app.lm.lm_utils import get_llm_client
from app.core.load_prompt import load_prompt

load_dotenv()


@dataclass
class RelevanceResult:
    """
    相关性判断结果
    score: 0~1 的浮点数，越高表示越相关
    is_relevant: 是否达到相关性阈值（≥CACHE_RELEVANCE_THRESHOLD）
    method: 使用的判断方法（embedding / cross_encoder / llm）
    reason: 判断理由描述
    """
    score: float
    is_relevant: bool
    method: str
    reason: str = ""


# ==================== 1级：BGE-M3 Embedding 余弦相似度（Bi-Encoder 快速预筛） ====================
# 参考：Reimers & Gurevych, "Sentence-BERT" (EMNLP 2019)
# 原理：将 query 和 cached_query 分别编码为向量，计算余弦相似度作为快速预筛
# 优点：速度快（O(1) 编码 + O(1) 余弦计算），适合大批量候选的初步过滤
# 缺点：无法捕捉 query 和 passage 之间的细粒度交互信息，精度低于 Cross-Encoder

def judge_relevance_embedding(query: str, cached_query: str) -> RelevanceResult:
    """
    使用 BGE-M3 双编码器（Bi-Encoder）计算 query 和 cached_query 的余弦相似度
    作为第一级快速预筛，阈值一般设较低（如 0.6），筛选出潜在相关候选
    参考：Sentence-BERT (Reimers & Gurevych, EMNLP 2019)

    参数：
        query: 用户新提问
        cached_query: 缓存中的历史问题摘要
    返回：
        RelevanceResult: 包含评分、是否相关、方法名和理由
    """
    try:
        logger.debug(f"[Embedding相关性] 正在编码: query='{query[:30]}...' vs cached='{cached_query[:30]}...'")

        # 使用 BGE-M3 生成双向量（稠密向量用于余弦相似度）
        embeddings = generate_embeddings([query, cached_query])
        dense_vecs = embeddings.get("dense")

        if not dense_vecs or len(dense_vecs) < 2:
            logger.warning("Embedding 生成失败，无法计算余弦相似度")
            return RelevanceResult(score=0.0, is_relevant=False, method="embedding", reason="向量生成失败")

        # 计算余弦相似度（归一化后的向量，内积即余弦相似度）
        import numpy as np
        vec_a = np.array(dense_vecs[0])
        vec_b = np.array(dense_vecs[1])
        cos_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8))

        # Embedding 预筛阈值
        threshold = 0.55
        is_rel = cos_sim >= threshold

        logger.debug(f"[Embedding相关性] 余弦相似度={cos_sim:.4f}, 阈值={threshold}, "
                      f"{'相关' if is_rel else '不相关'}")

        return RelevanceResult(
            score=round(cos_sim, 4),
            is_relevant=is_rel,
            method="embedding",
            reason=f"BGE-M3 Bi-Encoder 余弦相似度={cos_sim:.4f}（阈值={threshold}）"
        )

    except Exception as e:
        logger.error(f"[Embedding相关性] 计算失败: {str(e)}")
        return RelevanceResult(score=0.0, is_relevant=False, method="embedding", reason=f"计算异常: {str(e)}")


# ==================== 2级：BGE-Reranker 交叉编码器精确评分（Cross-Encoder） ====================
# 参考：Nogueira & Cho, "Passage Re-ranking with BERT" (2019) - Cross-Encoder 范式
# 参考：Xiao et al., "C-Pack: Resources for Chinese Embedding" (SIGIR 2023) - BGE-Reranker 实现
# 原理：将 query 和 passage 拼接后送入 Cross-Encoder 模型，直接输出相关性分数
#       相比 Bi-Encoder 的「分别编码→余弦相似度」，Cross-Encoder 能让 query 和 passage
#       在自注意力层中充分交互，精度显著提升，但速度较慢（需要联合前向传播）
# 优点：精度高，适合在 Embedding 预筛后对少量候选做精确评分
# 缺点：速度慢，不适合大批量候选的遍历评分

def judge_relevance_cross_encoder(query: str, cached_query: str, cached_answer: str = "") -> RelevanceResult:
    """
    使用 BGE-Reranker 交叉编码器（Cross-Encoder）精确判断语义相关性
    作为第二级精确评分，阈值使用全局配置（默认 0.85）
    参考：Nogueira & Cho, "Passage Re-ranking with BERT" (2019)
    参考：Xiao et al., "C-Pack" (SIGIR 2023)

    参数：
        query: 用户新提问
        cached_query: 缓存中的历史问题摘要
        cached_answer: 缓存中的历史答案摘要（增强上下文）
    返回：
        RelevanceResult: 包含评分、是否相关、方法名和理由
    """
    try:
        logger.debug(f"[Cross-Encoder相关性] 正在评分: query='{query[:30]}...' vs cached='{cached_query[:30]}...'")

        # 获取 BGE-Reranker 模型实例（项目已有该模型）
        reranker = get_reranker_model()
        if reranker is None:
            logger.warning("BGE-Reranker 模型不可用，跳过 Cross-Encoder 评分")
            return RelevanceResult(score=0.0, is_relevant=False, method="cross_encoder", reason="模型不可用")

        # 构建评分输入：将 query 与 cached_query+cached_answer 拼接作为 passage
        # 使用 cached_query 作为主候选，拼接 cached_answer 提供更多上下文
        passage = cached_query
        if cached_answer:
            passage = f"{cached_query} {cached_answer[:200]}"

        # BGE-Reranker compute_score 接受 List[Tuple[str, str]]
        # 格式: [(query, passage)]
        score = reranker.compute_score([[query, passage]])

        # 解析得分（返回列表或标量）
        if isinstance(score, (list, tuple)) and len(score) > 0:
            relevance_score = float(score[0])
        else:
            relevance_score = float(score)

        # 归一化到 0~1 区间（BGE-Reranker 原始得分可能是任意范围）
        # 使用 sigmoid 映射到 0~1
        relevance_score = 1.0 / (1.0 + pow(2.71828, -relevance_score))

        # 阈值判断（使用全局配置）
        threshold = redis_config.relevance_threshold
        is_rel = relevance_score >= threshold

        logger.debug(f"[Cross-Encoder相关性] 评分={relevance_score:.4f}, 阈值={threshold}, "
                      f"{'相关' if is_rel else '不相关'}")

        return RelevanceResult(
            score=round(relevance_score, 4),
            is_relevant=is_rel,
            method="cross_encoder",
            reason=f"BGE-Reranker Cross-Encoder 评分={relevance_score:.4f}（阈值={threshold}）"
        )

    except Exception as e:
        logger.error(f"[Cross-Encoder相关性] 评分失败: {str(e)}")
        # 降级：返回 0 分，等待上层使用 LLM 兜底
        return RelevanceResult(score=0.0, is_relevant=False, method="cross_encoder", reason=f"评分异常: {str(e)}")


# ==================== 3级：LLM 语义相关性判断（LLM-as-Judge 兜底） ====================
# 参考：Zheng et al., "Judging LLM-as-a-Judge" (NeurIPS 2023)
# 原理：利用大模型（如 qwen）的深度语义理解能力，判断两句问话是否在询问同一件事
# 优点：能处理复杂改写、指代消解、跨语言等场景，最灵活
# 缺点：速度最慢，成本最高，仅作为前两级均不可用时的兜底

def judge_relevance_llm(query: str, cached_query: str, cached_answer: str = "") -> RelevanceResult:
    """
    使用 LLM（LLM-as-Judge 范式）进行语义相关性判断
    作为第三级兜底，当 Cross-Encoder 和 Embedding 均不可用时使用
    参考：Zheng et al., "Judging LLM-as-a-Judge" (NeurIPS 2023)

    参数：
        query: 用户新提问
        cached_query: 缓存中的历史问题摘要
        cached_answer: 缓存中的历史答案摘要
    返回：
        RelevanceResult: 包含评分、是否相关、方法名和理由
    """
    try:
        logger.info(f"[LLM相关性] 调用LLM判断相关性: query='{query[:30]}...'")

        # 调用 LLM 进行相关性判断（使用已有的 get_llm_client，开启 JSON 模式）
        llm = get_llm_client(json_mode=True)

        # 加载相关性判断提示词
        prompt = load_prompt(
            "cache_relevance_check",
            user_query=query,
            cached_query=cached_query,
            cached_answer=cached_answer[:300] if cached_answer else ""
        )

        # 调用 LLM
        response = llm.invoke(prompt)
        content = response.content.strip()

        # 清理 Markdown 代码块标记
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        if content.startswith("```"):
            content = content.replace("```", "").strip()

        # 解析 JSON 结果
        result = json.loads(content)
        score = float(result.get("relevance_score", 0.0))
        is_relevant = bool(result.get("is_relevant", False))
        reason = result.get("reason", "LLM 判断完成")

        return RelevanceResult(
            score=round(score, 4),
            is_relevant=is_relevant,
            method="llm",
            reason=f"LLM-as-Judge: {reason}"
        )

    except json.JSONDecodeError as e:
        logger.error(f"[LLM相关性] 解析 LLM 输出失败: {e}, content={content[:100] if 'content' in dir() else 'N/A'}")
        return RelevanceResult(score=0.0, is_relevant=False, method="llm", reason="LLM 输出解析失败")
    except Exception as e:
        logger.error(f"[LLM相关性] LLM 判断失败: {str(e)}")
        return RelevanceResult(score=0.0, is_relevant=False, method="llm", reason=f"LLM 异常: {str(e)}")


# ==================== 综合判断入口（三级递进） ====================

def judge_relevance(
    query: str,
    cached_query: str,
    cached_answer: str = ""
) -> RelevanceResult:
    """
    三级递进式相关性判断（综合入口）
    策略：Embedding 快速预筛 → Cross-Encoder 精确评分 → LLM 兜底判断
    参考当前信息检索领域标准的「Retrieve → Re-rank → Judge」pipeline:
      1st stage: Bi-Encoder retrieval (Reimers & Gurevych, 2019)
      2nd stage: Cross-Encoder re-ranking (Nogueira & Cho, 2019; Xiao et al., 2023)
      3rd stage: LLM-as-Judge (Zheng et al., 2023)

    参数：
        query: 用户新提问
        cached_query: 缓存中的历史问题摘要
        cached_answer: 缓存中的历史答案摘要（用于增强 Cross-Encoder 和 LLM 判断）
    返回：
        RelevanceResult: 最终判断结果
    """
    threshold = redis_config.relevance_threshold

    # ===== 第1级：Embedding 快速预筛 =====
    # Bi-Encoder (Sentence-BERT 范式) —— 速度快但精度有限
    emb_result = judge_relevance_embedding(query, cached_query)
    if not emb_result.is_relevant:
        # 第一级已明确不相关，直接返回（避免后续耗时操作）
        logger.debug(f"[相关性判断] Embedding 预筛不相关 (score={emb_result.score:.4f}), 跳过后续判断")
        return emb_result

    # Embedding 预筛认为可能相关，进入第二级精确判断
    logger.debug(f"[相关性判断] Embedding 预筛通过 (score={emb_result.score:.4f}), 进入 Cross-Encoder 精判")

    # ===== 第2级：Cross-Encoder 精确评分 =====
    # BGE-Reranker (Nogueira & Cho, 2019; Xiao et al., 2023)
    ce_result = judge_relevance_cross_encoder(query, cached_query, cached_answer)
    if ce_result.score >= 0.0 and ce_result.method != "cross_encoder":
        # Cross-Encoder 不可用时，进入第3级 LLM 判断
        pass
    elif ce_result.is_relevant or ce_result.score >= threshold:
        # Cross-Encoder 明确相关，直接返回
        return ce_result
    elif ce_result.score > 0.3:
        # Cross-Encoder 评分在模糊区间（0.3~阈值），进入第3级 LLM 验证
        logger.debug(f"[相关性判断] Cross-Encoder 模糊区间 (score={ce_result.score:.4f}), 进入 LLM 终判")
    else:
        # Cross-Encoder 明确不相关
        return ce_result

    # ===== 第3级：LLM 兜底 =====
    # LLM-as-Judge (Zheng et al., 2023) —— 最慢但语义理解最强
    llm_result = judge_relevance_llm(query, cached_query, cached_answer)
    return llm_result


# ==================== 批量缓存相关性检测（用于遍历会话缓存摘要） ====================

def check_cache_relevance(
    user_query: str,
    cached_summaries: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Tuple[Dict[str, Any], RelevanceResult]]:
    """
    批量检测用户提问与多个缓存摘要的相关性，返回按相关性降序排列的 Top-K 结果
    用途：遍历会话的所有缓存摘要，找出与当前提问最相关的缓存

    参数：
        user_query: 用户新提问
        cached_summaries: 缓存摘要列表，每项含 query/answer/item_names 等字段
        top_k: 返回最相关的 Top-K 条
    返回：
        按相关性降序排列的 (摘要, 相关性结果) 列表
    """
    if not user_query or not cached_summaries:
        return []

    scored_results = []

    for summary in cached_summaries:
        cached_query = summary.get("query", "") or summary.get("summary_query", "")
        cached_answer = summary.get("answer", "") or summary.get("summary_answer", "")

        if not cached_query:
            continue

        # 执行三级递进相关性判断
        result = judge_relevance(user_query, cached_query, cached_answer)
        scored_results.append((summary, result))

    # 按相关性分数降序排列
    scored_results.sort(key=lambda x: x[1].score, reverse=True)

    # 返回 Top-K
    return scored_results[:top_k]


if __name__ == "__main__":
    """本地测试：验证三级递进式相关性判断功能"""
    logger.info("===== 相关性判断模块本地测试 =====")

    # 测试用例 1：高度相关（同一设备的不同问法）
    query_1 = "烫金机温度怎么调"
    cached_1 = "HAK 180 烫金机温度设置方法"
    answer_1 = "HAK 180 烫金机推荐设置温度为 110℃..."

    result_1 = judge_relevance(query_1, cached_1, answer_1)
    logger.info(f"测试1 [高度相关]: score={result_1.score:.4f}, method={result_1.method}, "
                f"relevant={result_1.is_relevant}, reason={result_1.reason}")

    # 测试用例 2：低度相关（不同设备）
    query_2 = "华为P60手机怎么开热点"
    cached_2 = "HAK 180 烫金机操作说明"

    result_2 = judge_relevance(query_2, cached_2)
    logger.info(f"测试2 [低度相关]: score={result_2.score:.4f}, method={result_2.method}, "
                f"relevant={result_2.is_relevant}, reason={result_2.reason}")

    # 测试用例 3：批量检测
    cached_list = [
        {"query": "HAK 180 烫金机怎么开机", "answer": "打开电源开关即可"},
        {"query": "烫金机温度设置指南", "answer": "建议设置在 110℃"},
        {"query": "华为手机充电慢怎么办", "answer": "建议更换原装充电器"}
    ]
    results = check_cache_relevance("烫金机温度调节步骤", cached_list)
    logger.info(f"测试3 [批量检测]: 最相关={results[0][0]['query'] if results else '无'}, "
                f"score={results[0][1].score:.4f}" if results else "无结果")

    logger.info("===== 相关性判断模块测试完成 =====")
