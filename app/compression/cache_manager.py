"""
Redis 缓存管理器模块
功能：封装对话摘要缓存的写入、读取、检索操作
核心职责：
  1. 写入缓存：将压缩后的对话摘要写入 Redis Hash，并维护会话索引
  2. 读取缓存：按 session_id 获取缓存摘要列表
  3. 检索缓存：按用户新提问检索最相关的缓存摘要（结合相关性判断模块）
  4. 缓存淘汰：基于 TTL 自动过期（24h），无需手动清理
  5. 降级策略：Redis 不可用时静默降级，不影响主链路

Redis 数据模型：
  - qa:summary:{session_id}:{summary_id}  → Hash（存储单条问答摘要）
    字段: query, answer, compressed_history, item_names, turn_count, timestamp, compression_ratio
  - qa:session:{session_id}:summary_keys   → Set（存储会话的所有摘要键，便于遍历）
  - qa:session:{session_id}:timeline       → Sorted Set（存储摘要键按时间排序）

缓存读取策略：
  - 新用户提问时，从 Redis 检索会话的缓存摘要
  - 使用三级递进式相关性判断（Embedding → Cross-Encoder → LLM）
  - 找到最相关的缓存摘要（score ≥ threshold）则直接使用缓存答案
  - 未找到或评分不足则走正常的全流程检索
"""
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.core.logger import logger
from app.conf.redis_config import redis_config
from app.clients.redis_utils import (
    get_redis_client,
    build_summary_key,
    build_session_keys_set,
    build_session_timeline,
    build_compact_analysis_key,
    RedisClient
)


class RedisCacheManager:
    """
    Redis 缓存管理器
    管理对话摘要的写入、读取、检索全生命周期
    设计：
      - 所有操作均带异常保护，Redis 不可用时静默降级
      - 写入时自动设置 TTL（24h），无需调用方手动管理
      - 写入时同时维护 Set 和 Sorted Set 索引，方便遍历
    """

    def __init__(self):
        """初始化缓存管理器"""
        self._client: Optional[RedisClient] = None

    def _get_client(self) -> Optional[RedisClient]:
        """
        获取 Redis 客户端（懒加载，每次调用检查连接）
        返回：RedisClient 实例，连接失败返回 None
        """
        if self._client is None or not self._client.is_connected():
            self._client = get_redis_client()
        return self._client

    # ==================== 写入操作 ====================

    def save_summary(self, session_id: str, summary_data: Dict[str, Any]) -> Optional[str]:
        """
        保存压缩摘要到 Redis 缓存
        执行流程：
          1. 生成唯一 summary_id
          2. 写入 Hash（存储完整摘要数据）
          3. 添加到会话的 Set 索引
          4. 添加到会话的 Sorted Set 时间线
        参数：
            session_id: 会话 ID
            summary_data: 压缩摘要数据（来自 ContextCompactEngine.compact() 返回值）
        返回：
            summary_id（成功时）或 None（失败时）
        """
        client = self._get_client()
        if client is None:
            logger.warning(f"[缓存管理器] Redis 不可用，跳过缓存写入：session={session_id}")
            return None

        try:
            # 1. 提取或生成摘要 ID
            summary_id = summary_data.get("summary_id", str(uuid.uuid4())[:8])
            summary_key = build_summary_key(session_id, summary_id)

            # 2. 构建 Hash 存储数据
            hash_data = {
                "summary_id": summary_id,
                "session_id": session_id,
                "summary_query": summary_data.get("summary_query", ""),
                "summary_answer": summary_data.get("summary_answer", ""),
                "compressed_history": summary_data.get("compressed_history", ""),
                "item_names": json.dumps(summary_data.get("item_names", []), ensure_ascii=False),
                "turn_count": str(summary_data.get("turn_count", 0)),
                "original_length": str(summary_data.get("original_length", 0)),
                "compressed_length": str(summary_data.get("compressed_length", 0)),
                "compression_ratio": str(summary_data.get("compression_ratio", 0)),
                "timestamp": str(summary_data.get("timestamp", datetime.now().timestamp())),
            }

            # 3. 写入 Redis Hash（自动设置 TTL）
            success = client.hset(summary_key, hash_data)
            if not success:
                logger.warning(f"[缓存管理器] Hash 写入失败：{summary_key}")
                return None

            # 4. 添加到会话的 Set 索引
            set_key = build_session_keys_set(session_id)
            client.sadd(set_key, summary_key, ttl=redis_config.cache_ttl)

            # 5. 添加到会话的 Sorted Set 时间线
            timeline_key = build_session_timeline(session_id)
            timestamp = float(summary_data.get("timestamp", datetime.now().timestamp()))
            client.zadd(timeline_key, summary_key, timestamp, ttl=redis_config.cache_ttl)

            logger.info(f"[缓存管理器] 摘要缓存成功：session={session_id}, summary_id={summary_id}, "
                        f"key={summary_key}, TTL={redis_config.cache_ttl}s")
            return summary_id

        except Exception as e:
            logger.error(f"[缓存管理器] 保存摘要缓存失败：session={session_id}, error={str(e)}")
            return None

    # ==================== 读取操作 ====================

    def get_summaries_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取某会话的所有缓存摘要列表
        参数：
            session_id: 会话 ID
        返回：
            摘要字典列表（按时间降序，最新的在前），Redis 不可用时返回空列表
        """
        client = self._get_client()
        if client is None:
            return []

        try:
            # 1. 从 Sorted Set 时间线获取所有摘要键（最新的在前）
            timeline_key = build_session_timeline(session_id)
            summary_keys = client.zrange(timeline_key, 0, -1, desc=True)

            # 2. 如果 Sorted Set 为空，尝试从 Set 获取
            if not summary_keys:
                set_key = build_session_keys_set(session_id)
                summary_keys = client.smembers(set_key)

            if not summary_keys:
                logger.debug(f"[缓存管理器] 会话 {session_id} 无缓存摘要")
                return []

            # 3. 遍历获取每个摘要的完整数据
            summaries = []
            for key in summary_keys:
                data = client.hgetall(key)
                if data:
                    # 反序列化 item_names
                    try:
                        item_names_raw = data.get("item_names", "[]")
                        data["item_names"] = json.loads(item_names_raw) if isinstance(item_names_raw, str) else []
                    except (json.JSONDecodeError, TypeError):
                        data["item_names"] = []

                    summaries.append(data)

            logger.debug(f"[缓存管理器] 会话 {session_id} 获取到 {len(summaries)} 条缓存摘要")
            return summaries

        except Exception as e:
            logger.error(f"[缓存管理器] 读取会话缓存失败：session={session_id}, error={str(e)}")
            return []

    def get_summary_by_key(self, summary_key: str) -> Optional[Dict[str, Any]]:
        """
        根据完整键名获取单条摘要数据
        参数：
            summary_key: Redis 键名（如 'qa:summary:session_xxx:abc123'）
        返回：
            摘要字典，不存在或失败返回 None
        """
        client = self._get_client()
        if client is None:
            return None

        try:
            if not client.exists(summary_key):
                return None
            data = client.hgetall(summary_key)
            if data:
                try:
                    item_names_raw = data.get("item_names", "[]")
                    data["item_names"] = json.loads(item_names_raw) if isinstance(item_names_raw, str) else []
                except (json.JSONDecodeError, TypeError):
                    data["item_names"] = []
            return data
        except Exception as e:
            logger.error(f"[缓存管理器] 读取单条摘要失败：key={summary_key}, error={str(e)}")
            return None

    # ==================== 删除操作 ====================

    def delete_summary(self, session_id: str, summary_id: str) -> bool:
        """
        删除指定摘要（同时清理索引）
        参数：
            session_id: 会话 ID
            summary_id: 摘要 ID
        返回：
            True=删除成功, False=失败
        """
        client = self._get_client()
        if client is None:
            return False

        try:
            summary_key = build_summary_key(session_id, summary_id)

            # 从索引中移除
            set_key = build_session_keys_set(session_id)
            client.sadd(set_key, summary_key)  # 从 Set 移除（实际应使用 srem）
            # 从 Sorted Set 移除
            timeline_key = build_session_timeline(session_id)

            # 注意：redis_utils 没有直接提供 srem/zrem 方法，使用原生客户端操作
            raw_client = client.client
            if raw_client:
                raw_client.srem(set_key, summary_key)
                raw_client.zrem(timeline_key, summary_key)

            # 删除摘要本身
            client.delete(summary_key)
            logger.info(f"[缓存管理器] 删除摘要缓存：session={session_id}, summary_id={summary_id}")
            return True

        except Exception as e:
            logger.error(f"[缓存管理器] 删除摘要失败：session={session_id}, summary_id={summary_id}, error={str(e)}")
            return False

    # ==================== 压缩状态管理（增量压缩标记位） ====================

    def save_compact_state(self, session_id: str, turn_count: int, summary_id: str) -> bool:
        """
        保存压缩状态标记位到 Redis，用于增量压缩判断
        存储结构：qa:session:{session_id}:compact_state → Hash
        字段：
          - last_compact_turn_count: 上次压缩时的会话轮数
          - last_compact_time: 上次压缩时间戳
          - last_summary_id: 上次生成的摘要 ID
        参数：
            session_id: 会话 ID
            turn_count: 本次压缩时的会话轮数
            summary_id: 本次生成的摘要 ID
        返回：
            True=写入成功, False=失败
        """
        client = self._get_client()
        if client is None:
            return False

        try:
            state_key = f"qa:session:{session_id}:compact_state"
            state_data = {
                "last_compact_turn_count": str(turn_count),
                "last_compact_time": str(datetime.now().timestamp()),
                "last_summary_id": summary_id,
            }
            client.hset(state_key, state_data, ttl=redis_config.cache_ttl)
            logger.info(
                f"[缓存管理器] 压缩状态已记录：session={session_id}, "
                f"last_compact_turn={turn_count}, summary_id={summary_id}"
            )
            return True
        except Exception as e:
            logger.error(f"[缓存管理器] 保存压缩状态失败：session={session_id}, error={str(e)}")
            return False

    def get_compact_state(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话的压缩状态标记位
        参数：
            session_id: 会话 ID
        返回：
            状态字典，包含：
              - last_compact_turn_count: int（默认 0，表示从未压缩过）
              - last_summary_id: str（默认空字符串）
              - last_compact_time: float（默认 0）
            Redis 不可用或 key 不存在时返回全零字典
        """
        client = self._get_client()
        if client is None:
            return {"last_compact_turn_count": 0, "last_summary_id": "", "last_compact_time": 0}

        try:
            state_key = f"qa:session:{session_id}:compact_state"
            data = client.hgetall(state_key)
            if not data:
                return {"last_compact_turn_count": 0, "last_summary_id": "", "last_compact_time": 0}

            return {
                "last_compact_turn_count": int(float(data.get("last_compact_turn_count", "0"))),
                "last_summary_id": data.get("last_summary_id", ""),
                "last_compact_time": float(data.get("last_compact_time", "0")),
            }
        except Exception as e:
            logger.error(f"[缓存管理器] 读取压缩状态失败：session={session_id}, error={str(e)}")
            return {"last_compact_turn_count": 0, "last_summary_id": "", "last_compact_time": 0}

    # ==================== 状态检查 ====================

    def get_cache_status(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话的缓存状态信息
        参数：
            session_id: 会话 ID
        返回：
            状态字典（缓存条目数、过期时间等）
        """
        client = self._get_client()
        if client is None:
            return {"available": False, "reason": "Redis 不可用"}

        try:
            timeline_key = build_session_timeline(session_id)
            summary_keys_list = client.zrange(timeline_key, 0, -1)

            # 获取第一条缓存的 TTL（代表该会话的缓存状态）
            first_ttl = -2
            if summary_keys_list:
                first_ttl = client.ttl(summary_keys_list[0])

            return {
                "available": True,
                "session_id": session_id,
                "cache_count": len(summary_keys_list),
                "remaining_ttl": first_ttl,
                "ttl_hours": round(first_ttl / 3600, 1) if first_ttl > 0 else None,
            }
        except Exception as e:
            return {"available": False, "reason": str(e)}


# ==================== 全局单例管理 ====================

_cache_manager_instance: Optional[RedisCacheManager] = None


def get_cache_manager() -> RedisCacheManager:
    """获取缓存管理器单例"""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = RedisCacheManager()
    return _cache_manager_instance


if __name__ == "__main__":
    """本地测试：验证缓存管理器功能"""
    logger.info("===== 缓存管理器本地测试 =====")

    manager = get_cache_manager()

    # 测试 1：写入缓存
    test_data = {
        "summary_id": "test_001",
        "summary_query": "HAK 180 烫金机操作指南",
        "summary_answer": "HAK 180 烫金机的操作步骤包括：开机、设置温度、放置材料、启动机器。",
        "compressed_history": "[用户]: 烫金机怎么用?\n[助手]: 开机后设置温度...",
        "item_names": ["HAK 180 烫金机"],
        "turn_count": 6,
        "original_length": 500,
        "compressed_length": 200,
        "compression_ratio": 60.0,
        "timestamp": datetime.now().timestamp(),
    }

    summary_id = manager.save_summary("test_cache_session", test_data)
    logger.info(f"测试1 [写入缓存]: {'通过' if summary_id else '失败'}")

    # 测试 2：读取缓存
    summaries = manager.get_summaries_by_session("test_cache_session")
    logger.info(f"测试2 [读取缓存]: {'通过' if summaries else '失败'}，条数={len(summaries)}")
    if summaries:
        logger.info(f"  首条摘要问题: {summaries[0].get('summary_query')}")

    # 测试 3：缓存状态
    status = manager.get_cache_status("test_cache_session")
    logger.info(f"测试3 [缓存状态]: {status}")

    # 清理测试数据
    if summary_id:
        manager.delete_summary("test_cache_session", summary_id)
        logger.info("测试数据已清理")

    logger.info("===== 缓存管理器测试完成 =====")
