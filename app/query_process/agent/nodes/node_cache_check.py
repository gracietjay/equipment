"""
Redis 缓存检查节点

命中：摘要当导航（compressed_history + item_names），走快路径
未命中：加载 MongoDB 长期记忆（history_item_names），走增强路径
"""
import sys
import json
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.compression.cache_manager import get_cache_manager, RedisCacheManager
from app.compression.context_compact_engine import get_compact_engine, ContextCompactEngine
from app.compression.relevance_judger import check_cache_relevance, RelevanceResult
from app.clients.mongo_history_utils import get_recent_messages

load_dotenv()

CACHE_NAVIGATION_THRESHOLD = 0.70

# 长期记忆实体提取窗口：取最近 N 轮用户发言的 item_names，旧实体随话题切换自然淘汰
RECENT_ROUND_WINDOW = 3  # 最近 3 轮用户发言


def _load_long_term_memory(state, session_id: str):
    """从 MongoDB 近期历史中提取 item_names，缓存未命中时辅助检索。

    按「用户发言轮次」做滑动窗口：只提取最近 RECENT_ROUND_WINDOW 轮用户消息中
    携带的 item_names，而不是按原始消息条数截取。避免多轮话题切换后旧实体
    仍混入 Milvus 过滤条件引入噪声。
    """
    history_item_names = []
    try:
        history = state.get("history", [])
        if not history:
            # 多取一些原始消息以确保能覆盖 N 轮用户发言
            history = get_recent_messages(session_id, limit=20)
        # 只取 user 角色的消息，取最近 N 轮
        user_messages = [msg for msg in history if msg.get("role") == "user"]
        recent_user_rounds = user_messages[-RECENT_ROUND_WINDOW:]
        history_item_names = list(set(
            n for msg in recent_user_rounds
            for n in (msg.get("item_names") or []) if n and str(n).strip()
        ))
        if history_item_names:
            logger.info(
                f"MongoDB 长期记忆提取到 item_names "
                f"(滑动窗口={RECENT_ROUND_WINDOW}轮用户发言, "
                f"实际命中={len(recent_user_rounds)}条user消息): {history_item_names}"
            )
        else:
            logger.debug("MongoDB 长期记忆滑动窗口内未提取到 item_names")
    except Exception as e:
        logger.warning(f"MongoDB 长期记忆读取失败: {e}")
    state["history_item_names"] = history_item_names


def step_1_check_cache(
    session_id: str,
    user_query: str,
    cache_manager: RedisCacheManager
) -> List[Dict[str, Any]]:
    """从 Redis 获取当前会话的缓存摘要列表"""
    try:
        summaries = cache_manager.get_summaries_by_session(session_id)
        logger.info(f"[缓存检查] 获取到 {len(summaries)} 条缓存摘要")
        return summaries
    except Exception as e:
        logger.error(f"[缓存检查] 获取缓存失败: {str(e)}")
        return []


def step_2_evaluate_relevance(
    user_query: str,
    cached_summaries: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """判断用户提问与缓存摘要的相关性，返回最匹配的摘要或 None"""
    if not user_query or not cached_summaries:
        return None

    try:
        scored_results = check_cache_relevance(user_query, cached_summaries, top_k=1)

        if not scored_results:
            return None

        best_summary, relevance = scored_results[0]
        threshold = CACHE_NAVIGATION_THRESHOLD

        if relevance.is_relevant and relevance.score >= threshold:
            logger.info(f"[缓存检查] 命中 score={relevance.score:.4f} "
                        f"query='{best_summary.get('summary_query', '')[:50]}...'")
            return best_summary
        else:
            logger.info(f"[缓存检查] 未命中 score={relevance.score:.4f} < {threshold}")
            return None

    except Exception as e:
        logger.error(f"[缓存检查] 步骤2: 相关性判断异常: {str(e)}")
        return None


def step_3_trigger_compact(
    session_id: str,
    history: List[Dict[str, Any]],
    compact_engine: ContextCompactEngine,
    cache_manager: RedisCacheManager
) -> None:
    """检查是否需要触发历史压缩（首次>=5轮全量 / 增量>=3轮仅压缩新增并拼接旧摘要）"""
    try:
        # 1. 读取上次压缩状态（从未压缩过则返回 turn_count=0）
        compact_state = cache_manager.get_compact_state(session_id)
        last_compact_turn_count = compact_state.get("last_compact_turn_count", 0)
        last_summary_id = compact_state.get("last_summary_id", "")

        # 2. 增量判断：首次 >=5 轮触发，之后每新增 >=3 轮再触发
        if compact_engine.should_compact(history, last_compact_turn_count):
            # 3. 如果是增量压缩，从 Redis 读取上次的 compressed_history
            previous_compressed_history = ""
            if last_compact_turn_count > 0 and last_summary_id:
                prev_summary = cache_manager.get_summary_by_key(
                    f"qa:summary:{session_id}:{last_summary_id}"
                )
                if prev_summary:
                    previous_compressed_history = prev_summary.get("compressed_history", "")
                    logger.info(
                        f"[缓存检查] 读取到上次摘要 compressed_history，"
                        f"长度={len(previous_compressed_history)}，summary_id={last_summary_id}"
                    )

            logger.info(
                f"[缓存检查] 触发历史压缩（会话 {session_id}），"
                f"上次压缩轮数={last_compact_turn_count}，"
                f"模式={'增量' if previous_compressed_history else '全量（首次）'}"
            )
            compact_result = compact_engine.compact(
                session_id=session_id,
                history=history,
                force=False,
                last_compact_turn_count=last_compact_turn_count,
                previous_compressed_history=previous_compressed_history
            )
            if compact_result:
                summary_id = cache_manager.save_summary(session_id, compact_result)
                if summary_id:
                    # 4. 记录压缩状态标记位，供下次增量判断
                    turn_count = compact_result.get("turn_count", 0)
                    cache_manager.save_compact_state(session_id, turn_count, summary_id)
                    logger.info(
                        f"[缓存检查] 压缩完成并记录状态，summary_id={summary_id}, "
                        f"turn_count={turn_count}"
                    )
    except Exception as e:
        logger.error(f"[缓存检查] 压缩/缓存异常: {str(e)}")


def node_cache_check(state: QueryGraphState) -> QueryGraphState:
    """
    缓存检查主节点

    命中：设 cache_hit=True, compressed_history, item_names → 快路径检索
    未命中：设 cache_hit=False, 加载 MongoDB 长期记忆 → 增强路径检索
    """
    logger.info("--- node_cache_check (缓存检查) 节点开始处理 ---")
    session_id = state.get("session_id", "")
    user_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    # 初始化组件
    cache_manager = get_cache_manager()
    compact_engine = get_compact_engine()

    # ===== 步骤1：检查 Redis 缓存 =====
    cached_summaries = step_1_check_cache(session_id, user_query, cache_manager)

    # ===== 步骤2（可选）：获取历史并尝试触发压缩 =====
    # 从 state 中获取历史（上游可能在入口处已注入）
    history = state.get("history", [])
    if history:
        step_3_trigger_compact(session_id, history, compact_engine, cache_manager)

    # ===== 步骤3：判断最相关缓存 =====
    if cached_summaries:
        best_cache = step_2_evaluate_relevance(user_query, cached_summaries)
        if best_cache:
            # 命中：摘要当导航，不设 answer，走快路径
            compressed_history = best_cache.get("compressed_history", "")
            item_names = best_cache.get("item_names", [])
            cached_query = best_cache.get("summary_query", "")

            if isinstance(item_names, str):
                try:
                    item_names = json.loads(item_names)
                except (json.JSONDecodeError, TypeError):
                    item_names = []

            logger.info(f"[缓存检查] 命中，进入快路径: "
                        f"cached_query='{cached_query[:50]}...', "
                        f"item_names={item_names}, "
                        f"compressed_history_len={len(compressed_history)}")

            state["compressed_history"] = compressed_history
            state["item_names"] = item_names if isinstance(item_names, list) else []
            state["cache_hit"] = True
        else:
            # 未命中：加载 MongoDB 长期记忆辅助检索
            logger.info("[缓存检查] 未命中，加载长期记忆")
            state["cache_hit"] = False
            _load_long_term_memory(state, session_id)
    else:
        logger.info("[缓存检查] 无缓存，加载长期记忆")
        state["cache_hit"] = False
        _load_long_term_memory(state, session_id)

    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    logger.info("--- node_cache_check 节点处理结束 ---")

    return state


if __name__ == "__main__":
    """本地测试：验证缓存检查节点逻辑"""
    logger.info("===== 缓存检查节点本地测试 =====")

    # 测试用例1：无缓存（首次提问）
    state1 = {
        "session_id": "test_cache_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "history": [],
        "is_stream": False,
    }
    result1 = node_cache_check(state1)
    logger.info(f"测试1 [无缓存]: answer={result1.get('answer')}, from_cache={result1.get('from_cache', False)}")

    # 测试用例2：有历史但不够压缩阈值
    history2 = [
        {"role": "user", "text": "你好"},
        {"role": "assistant", "text": "你好！有什么可以帮助您的？"}
    ]
    state2 = {
        "session_id": "test_cache_session_002",
        "original_query": "温度怎么设置？",
        "history": history2,
        "is_stream": False,
    }
    result2 = node_cache_check(state2)
    logger.info(f"测试2 [有历史-未达阈值]: answer={result2.get('answer')}, from_cache={result2.get('from_cache', False)}")

    logger.info("===== 缓存检查节点测试完成 =====")
