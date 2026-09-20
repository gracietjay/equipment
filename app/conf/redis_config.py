"""
Redis 缓存配置模块
功能：集中管理 Redis 相关配置，遵循项目统一的 dataclass + .env 风格
支持：记忆存储/会话摘要缓存的连接参数、TTL、相关性阈值等
"""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件，确保 os.getenv 能获取到 Redis 相关配置
load_dotenv()


@dataclass
class RedisConfig:
    """Redis 服务配置类（与 LLMConfig/MilvusConfig 风格一致）"""
    host: str               # Redis 主机地址
    port: int               # Redis 端口
    db: int                 # Redis 数据库编号
    password: str           # Redis 密码（无密码为空字符串）
    cache_ttl: int          # 缓存过期时间（秒），默认 24h
    relevance_threshold: float  # 缓存相关性判断阈值（BGE-Reranker 交叉编码器评分）
    compact_turn_threshold: int  # 触发历史压缩的会话轮数阈值（首次触发）
    compact_incremental_threshold: int  # 增量压缩阈值：上次压缩后新增多少轮才再次触发


# 实例化配置对象，自动从 .env 读取并绑定
redis_config = RedisConfig(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD", ""),
    cache_ttl=int(os.getenv("REDIS_CACHE_TTL", "86400")),
    relevance_threshold=float(os.getenv("CACHE_RELEVANCE_THRESHOLD", "0.85")),
    compact_turn_threshold=int(os.getenv("COMPACT_TURN_THRESHOLD", "5")),
    compact_incremental_threshold=int(os.getenv("COMPACT_INCREMENTAL_THRESHOLD", "3"))
)
