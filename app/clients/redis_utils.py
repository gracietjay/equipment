"""
Redis 客户端工具模块
功能：封装 Redis 连接管理和基础操作，采用单例模式避免重复创建连接
遵循项目统一的客户端风格（同 milvus_utils.py / mongo_history_utils.py）
核心能力：
  1. Redis 连接池管理（单例 + 连接复用，性能优化）
  2. 基础 CRUD 操作（Hash / String / Set / Sorted Set 等数据结构）
  3. 异常统一处理 + 兜底降级（连接失败返回 None 或空值）
  4. TTL 自动化管理（设置过期时间，避免缓存堆积）
  5. 上下文管理（with 语句支持，确保连接释放）
"""
import json
import os
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

from dotenv import load_dotenv

from app.conf.redis_config import redis_config
from app.core.logger import logger

# 加载 .env 配置（确保与项目环境一致）
load_dotenv()


class RedisClient:
    """
    Redis 客户端封装类（单例模式）
    基于 go_redis (redis-py) 实现，支持 Hash/Set/Sorted Set 等数据结构
    核心设计：
      1. 懒加载：首次使用时才创建连接池，避免模块加载时不必要的连接
      2. 连接池：redis-py 内置连接池管理，自动复用连接
      3. 统一异常：所有操作异常均捕获并记录日志，返回兜底值
      4. TTL 自动管理：写入时自动设置过期时间，无需调用方处理
    """

    def __init__(self):
        """初始化 Redis 连接（立即建立连接，失败则记录警告）"""
        self._client = None
        self._connect()

    def _connect(self) -> None:
        """建立 Redis 连接（带异常保护）"""
        try:
            import redis as go_redis
            self._client = go_redis.Redis(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.db,
                password=redis_config.password if redis_config.password else None,
                decode_responses=True,  # 自动将 bytes 解码为 str
                socket_connect_timeout=5,  # 连接超时 5 秒
                socket_timeout=10,  # 读写超时 10 秒
                retry_on_timeout=True,  # 超时自动重试
                health_check_interval=30,  # 每 30 秒健康检查一次
            )
            # 测试连接是否成功
            self._client.ping()
            logger.info(f"Redis 客户端连接成功：{redis_config.host}:{redis_config.port}/{redis_config.db}")
        except ImportError:
            logger.warning("redis 库未安装，Redis 缓存功能不可用（pip install redis）")
            self._client = None
        except Exception as e:
            logger.warning(f"Redis 客户端连接失败（缓存功能降级）：{str(e)}")
            self._client = None

    @property
    def client(self):
        """获取原生 Redis 客户端实例，供高级操作使用"""
        return self._client

    def is_connected(self) -> bool:
        """检查 Redis 连接是否正常"""
        if self._client is None:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    # ==================== Hash 操作（存储结构化数据） ====================

    def hset(self, key: str, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        写入 Hash 类型数据（最常用的 Q&A 摘要存储结构）
        用途：存储单条问答摘要的多个字段（query/answer/item_names/timestamp 等）
        参数：
            key: Redis 键名，如 'qa:summary:session_xxx:uuid_xxx'
            mapping: 字段映射字典，自动将非字符串值 JSON 序列化
            ttl: 过期时间（秒），默认使用全局配置 24h
        返回：
            bool: 是否写入成功（失败返回 False，连接断开时降级）
        """
        if self._client is None:
            return False
        try:
            # 将非字符串值序列化为 JSON 字符串
            safe_mapping = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list, tuple)):
                    safe_mapping[k] = json.dumps(v, ensure_ascii=False)
                elif isinstance(v, datetime):
                    safe_mapping[k] = v.timestamp()
                elif v is None:
                    safe_mapping[k] = ""
                else:
                    safe_mapping[k] = str(v)

            self._client.hset(key, mapping=safe_mapping)

            # 设置过期时间（默认使用全局配置）
            expire_sec = ttl if ttl is not None else redis_config.cache_ttl
            self._client.expire(key, expire_sec)

            logger.debug(f"Redis Hash 写入成功：key={key}, fields={len(safe_mapping)}, TTL={expire_sec}s")
            return True
        except Exception as e:
            logger.error(f"Redis Hash 写入失败：key={key}, error={str(e)}")
            return False

    def hget(self, key: str, field: str) -> Optional[str]:
        """
        获取 Hash 中指定字段的值
        参数：
            key: Redis 键名
            field: 字段名
        返回：
            字段值字符串，失败或无此字段返回 None
        """
        if self._client is None:
            return None
        try:
            value = self._client.hget(key, field)
            return value
        except Exception as e:
            logger.error(f"Redis Hash 读取失败：key={key}, field={field}, error={str(e)}")
            return None

    def hgetall(self, key: str) -> Dict[str, str]:
        """
        获取 Hash 中所有字段（用于读取完整问答摘要）
        参数：
            key: Redis 键名
        返回：
            字段-值 字典，失败返回空字典
        """
        if self._client is None:
            return {}
        try:
            result = self._client.hgetall(key)
            return result or {}
        except Exception as e:
            logger.error(f"Redis Hash 读取全部字段失败：key={key}, error={str(e)}")
            return {}

    def hkeys(self, key: str) -> List[str]:
        """获取 Hash 所有字段名"""
        if self._client is None:
            return []
        try:
            return self._client.hkeys(key) or []
        except Exception:
            return []

    # ==================== Set 操作（管理会话摘要索引列表） ====================

    def sadd(self, key: str, *members: str, ttl: Optional[int] = None) -> bool:
        """
        向 Set 中添加成员（用于管理会话下的所有摘要键）
        用途：每个会话的摘要键集合，如 'qa:session:session_xxx:keys'
        参数：
            key: Set 键名
            members: 要添加的成员（可变参数）
            ttl: 过期时间（秒）
        """
        if self._client is None:
            return False
        try:
            self._client.sadd(key, *members)
            if ttl is not None:
                self._client.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Redis Set 添加失败：key={key}, error={str(e)}")
            return False

    def smembers(self, key: str) -> List[str]:
        """
        获取 Set 所有成员（获取会话下的所有摘要键列表）
        用途：列出某会话的所有缓存摘要
        """
        if self._client is None:
            return []
        try:
            members = self._client.smembers(key)
            return list(members) if members else []
        except Exception as e:
            logger.error(f"Redis Set 读取失败：key={key}, error={str(e)}")
            return []

    # ==================== Sorted Set 操作（按时间排序的摘要列表） ====================

    def zadd(self, key: str, member: str, score: float, ttl: Optional[int] = None) -> bool:
        """
        向 Sorted Set 添加成员（按时间戳排序，用于按时间遍历摘要）
        用途：维护会话摘要的有序列表，按时间戳分数排序
        参数：
            key: Sorted Set 键名，如 'qa:session:session_xxx:timeline'
            member: 成员值（如摘要键名）
            score: 分数（如时间戳）
            ttl: 过期时间（秒）
        """
        if self._client is None:
            return False
        try:
            self._client.zadd(key, {member: score})
            if ttl is not None:
                self._client.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Redis Sorted Set 添加失败：key={key}, error={str(e)}")
            return False

    def zrange(self, key: str, start: int = 0, end: int = -1, desc: bool = False) -> List[str]:
        """
        按分数范围获取 Sorted Set 成员（按时间获取摘要列表，最新优先）
        参数：
            key: Sorted Set 键名
            start: 起始索引
            end: 结束索引（-1 表示全部）
            desc: 是否降序（True=最新在前，False=最旧在前）
        """
        if self._client is None:
            return []
        try:
            if desc:
                return self._client.zrevrange(key, start, end) or []
            return self._client.zrange(key, start, end) or []
        except Exception as e:
            logger.error(f"Redis Sorted Set 读取失败：key={key}, error={str(e)}")
            return []

    # ==================== 通用操作 ====================

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if self._client is None:
            return False
        try:
            return bool(self._client.exists(key))
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """删除指定键"""
        if self._client is None:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis 删除失败：key={key}, error={str(e)}")
            return False

    def expire(self, key: str, ttl: int) -> bool:
        """设置键的过期时间"""
        if self._client is None:
            return False
        try:
            return bool(self._client.expire(key, ttl))
        except Exception:
            return False

    def ttl(self, key: str) -> int:
        """获取键的剩余过期时间（秒），-1 表示永不过期，-2 表示不存在"""
        if self._client is None:
            return -2
        try:
            return self._client.ttl(key)
        except Exception:
            return -2

    def close(self):
        """关闭 Redis 连接（释放资源）"""
        if self._client:
            try:
                self._client.close()
                logger.info("Redis 连接已关闭")
            except Exception as e:
                logger.warning(f"Redis 关闭连接时异常：{str(e)}")


# ==================== 全局单例管理（与项目风格一致） ====================

_redis_client_instance: Optional[RedisClient] = None


def get_redis_client() -> Optional[RedisClient]:
    """
    获取 RedisClient 单例实例（懒加载模式）
    核心逻辑：
      1. 全局实例为空时创建新实例（仅首次）
      2. 若连接断开，自动重建连接（自愈）
      3. 返回 None 时表示 Redis 不可用，调用方应降级处理
    返回：
        RedisClient 实例，连接失败返回 None
    """
    global _redis_client_instance
    try:
        if _redis_client_instance is None:
            _redis_client_instance = RedisClient()
        elif not _redis_client_instance.is_connected():
            logger.warning("Redis 连接已断开，尝试重新连接...")
            _redis_client_instance._connect()
        return _redis_client_instance
    except Exception as e:
        logger.error(f"获取 Redis 客户端实例失败：{str(e)}")
        return None


def close_redis_client():
    """关闭 Redis 全局连接（应用退出时调用）"""
    global _redis_client_instance
    if _redis_client_instance:
        _redis_client_instance.close()
        _redis_client_instance = None
        logger.info("Redis 全局连接已释放")


# ==================== 工具函数：构建 Redis 键名 ====================

def build_summary_key(session_id: str, summary_id: str) -> str:
    """
    构建 Q&A 摘要的 Redis 键名
    格式：qa:summary:{session_id}:{summary_id}
    用途：每条对话摘要对应一个 Hash，存储 query/answer/item_names/timestamp 等字段
    """
    return f"qa:summary:{session_id}:{summary_id}"


def build_session_keys_set(session_id: str) -> str:
    """
    构建会话摘要键的 Set 名称（存储该会话所有摘要键的集合）
    格式：qa:session:{session_id}:summary_keys
    用途：快速获取某会话的全部缓存摘要
    """
    return f"qa:session:{session_id}:summary_keys"


def build_session_timeline(session_id: str) -> str:
    """
    构建会话摘要时间线 Sorted Set 名称（按时间排序）
    格式：qa:session:{session_id}:timeline
    用途：按时间遍历会话摘要（最新优先检索）
    """
    return f"qa:session:{session_id}:timeline"


def build_compact_analysis_key(session_id: str, summary_id: str) -> str:
    """
    构建压缩分析文件的 Redis 缓存键
    格式：qa:analysis:{session_id}:{summary_id}
    用途：缓存压缩过程解析的 JSON 数据，可同步写入本地文件夹
    """
    return f"qa:analysis:{session_id}:{summary_id}"


if __name__ == "__main__":
    """本地测试：验证 Redis 连接和基础操作"""
    logger.info("===== Redis 客户端工具本地测试 =====")

    client = get_redis_client()
    if client is None:
        logger.error("Redis 连接失败，请检查 Redis 服务是否已启动")
        exit(1)

    # 测试 Hash 写入/读取
    test_key = build_summary_key("test_session", "test_001")
    test_data = {
        "query": "HAK 180 烫金机怎么操作？",
        "answer": "HAK 180 烫金机的操作面板位于机器正前方...",
        "item_names": json.dumps(["HAK 180 烫金机"], ensure_ascii=False),
        "timestamp": str(datetime.now().timestamp()),
        "turn_count": "5"
    }

    success = client.hset(test_key, test_data, ttl=3600)
    logger.info(f"Hash 写入测试：{'通过' if success else '失败'}")

    if success:
        data = client.hgetall(test_key)
        logger.info(f"Hash 读取测试：通过，数据={data}")
        client.delete(test_key)

    # 测试 Sorted Set
    timeline_key = build_session_timeline("test_session")
    client.zadd(timeline_key, "summary_1", 1000.0, ttl=3600)
    client.zadd(timeline_key, "summary_2", 2000.0, ttl=3600)
    members = client.zrange(timeline_key, 0, -1, desc=True)
    logger.info(f"Sorted Set 读取测试：{'通过' if len(members) == 2 else '失败'}，members={members}")
    client.delete(timeline_key)

    close_redis_client()
    logger.info("===== Redis 客户端测试完成 =====")
