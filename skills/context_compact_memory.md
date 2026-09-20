---
name: context-compact-memory
description: 长期记忆压缩Skill — 对MongoDB中存储的会话历史进行渐进式摘要压缩，缓存到Redis并生成压缩过程解析文件
metadata:
  type: skill
  trigger: 会话超过5轮（由 COMPACT_TURN_THRESHOLD=5 控制），或用户明确要求「压缩此对话」
  scope: knowledge_base/app/compression/ 模块
  dependencies: Redis, MongoDB, LLM (qwen-flash), BGE-Reranker
---

# Context Compact Memory 技能

## 1. 技能概述

本技能实现了对历史对话的「渐进式披露」长期记忆压缩机制。采用懒加载策略——仅在需要时（触发条件满足时）才加载压缩流程，避免对系统主链路的性能影响。

核心能力：
- **短期记忆**（Redis）：缓存最近N轮对话的问答摘要，TTL=24h，用于快速响应重复问题
- **长期记忆**（MongoDB + 压缩）：超过5轮后对历史进行语义压缩，生成紧凑摘要，减少检索压力
- **三级递进相关性判断**：Embedding → Cross-Encoder → LLM，精准判断新提问与缓存的语义相关性

### 1.1 使用边界

| 维度 | 边界说明 |
|------|---------|
| **触发条件** | 会话轮数 ≥ COMPACT_TURN_THRESHOLD（默认5轮），或强制触发 |
| **缓存对象** | 仅压缩 user+assistant 的消息对，系统消息/元数据不参与压缩 |
| **缓存范围** | 按 session_id 隔离，不同会话的缓存互不干扰 |
| **缓存有效期** | TTL=86400秒（24小时），到期自动失效 |
| **缓存容量** | 单会话缓存数量无硬限制，但遍历全部缓存时通过三级递进方法确保效率 |
| **降级策略** | Redis不可用时静默降级，不阻塞主流程；压缩失败时保留原文全文 |

### 1.2 触发条件

1. **自动触发**：会话user消息数 ≥ COMPACT_TURN_THRESHOLD（默认5轮）
2. **手动触发**：用户明确要求「压缩对话」「汇总历史」「整理记忆」
3. **查询触发**：用户提问时，自动检查Redis缓存判断是否有相关可用摘要

---

## 2. Input：数据约束与格式校验

### 2.1 输入格式要求

压缩引擎期望的输入数据格式如下：

```python
# 历史对话记录（来自 MongoDB chat_message 集合）
[
    {
        "session_id": str,           # 会话唯一标识（必填）
        "role": str,                 # "user" 或 "assistant"（必填）
        "text": str,                 # 对话内容（必填，非空）
        "rewritten_query": str,      # 改写后的问题（可选，默认""）
        "item_names": List[str],     # 关联商品名（可选，默认[]）
        "ts": float,                 # 时间戳（可选，默认当前时间）
    },
    # ... 更多消息
]
```

### 2.2 格式校验与异常兜底

输入数据经过以下严格校验，不符合条件时进行兜底处理：

```
校验规则1：session_id 非空
  → 失败：日志警告，返回 None，跳过压缩

校验规则2：至少有一条 role="user" 的消息
  → 失败：日志警告，返回 None（无需压缩无对话的会话）

校验规则3：每条消息至少包含 role 和 text 字段
  → 失败：过滤掉无效消息，保留有效消息继续处理

校验规则4：text 字段非空字符串
  → 失败：跳过空文本消息，日志记录被跳过的消息数量

校验规则5：轮数检测（自动触发时）
  → 失败：user消息数 < 5轮 → 不触发压缩，等待后续轮次
```

### 2.3 代码实现

参考 `app/compression/context_compact_engine.py` 中的 `should_compact()` 和 `compact()` 方法。

---

## 3. 中间流程：压缩什么？怎样进行压缩？

### 3.1 压缩什么？

**要压缩的内容：**
- 用户历史提问（保留核心问题语义，去除重复/冗余表述）
- 助手的回答（保留关键信息，去除寒暄/模板化语言）
- 涉及的商品名/实体名（确保后续检索的精准性）

**不压缩（保留原始）的内容：**
- 系统级别的元数据（task_id, session_id 等）
- 时间戳、状态标记
- 错误信息（用于排查）
- 图片URL（单独处理）

**压缩原则：**
- 保留对话的核心脉络和上下文连贯性
- 去除寒暄、重复表达和非核心细节
- 保持关键信息的准确性（不编造不存在的事实）
- 压缩比控制在原内容的30%~50%

### 3.2 怎样进行压缩？（完整流程）

```
┌──────────────────────────────────────────────────────────┐
│                   压缩流程（6步）                          │
├──────────────────────────────────────────────────────────┤
│  步骤1：轮数检测                                          │
│  → 检查 user_count ≥ 5                                   │
│  → 不满足则跳过（等待下次触发）                             │
├──────────────────────────────────────────────────────────┤
│  步骤2：历史提取                                          │
│  → 从MongoDB获取最近50条记录                               │
│  → 按 ts 升序排列（从旧到新）                               │
├──────────────────────────────────────────────────────────┤
│  步骤3：格式化                                            │
│  → 转为 [用户]/[助手] 交替文本格式                          │
│  → 统计总轮数和原始文本长度                                  │
├──────────────────────────────────────────────────────────┤
│  步骤4：LLM语义压缩                                       │
│  → 使用 prompts/context_compact.prompt                    │
│  → 调用 qwen-flash 生成摘要                                │
│  → 目标：保留核心Q&A，压缩至30%~50%                        │
├──────────────────────────────────────────────────────────┤
│  步骤5：结果提取                                          │
│  → 提取核心问题摘要（summary_query）                        │
│  → 提取核心答案摘要（summary_answer）                       │
│  → 计算压缩比等统计信息                                     │
├──────────────────────────────────────────────────────────┤
│  步骤6：双路输出                                          │
│  → 输出1：缓存到 Redis（Hash + Set + Sorted Set）          │
│  → 输出2：生成解析文件到 context_compact/analysis/          │
└──────────────────────────────────────────────────────────┘
```

### 3.3 压缩策略详解

**语义压缩策略（LLM 层）：**
- **保留核心语义**：提取用户核心问题与助手关键回答
- **去除冗余**：过滤寒暄、重复、模板化语言
- **保持脉络**：按轮次顺序压缩，保留上下文连贯性
- **实体完整性**：记录涉及的商品名/实体名
- **比例控制**：压缩后长度控制在原始长度的30%~50%

**缓存检索策略（Redis 层）：**
1. 用户新提问时，从Redis获取会话的缓存摘要列表
2. 使用三级递进式相关性判断：
   - Stage 1: **BGE-M3 Embedding余弦相似度**（Bi-Encoder快速预筛）
   - Stage 2: **BGE-Reranker交叉编码器评分**（Cross-Encoder精确评分）
   - Stage 3: **LLM语义判断兜底**（LLM-as-Judge）
3. 最佳匹配评分 ≥ CACHE_RELEVANCE_THRESHOLD（默认0.85）= 缓存命中
4. 命中后直接返回缓存答案，跳过检索-重排-生成全流程

---

## 4. Output：双路输出

### 4.1 输出1：Redis 缓存

```redis
# Hash：单条问答摘要
qa:summary:{session_id}:{summary_id}
  ├── summary_query: "HAK 180 烫金机操作指南"          # 核心问题摘要
  ├── summary_answer: "HAK 180 烫金机的操作步骤包括..." # 核心答案摘要
  ├── compressed_history: "[用户]:...\n[助手]:..."    # 压缩后的完整历史
  ├── item_names: '["HAK 180 烫金机"]'                # 涉及商品名（JSON序列化）
  ├── turn_count: "6"                                  # 原始对话轮数
  ├── original_length: "500"                           # 原始文本长度
  ├── compressed_length: "200"                         # 压缩后长度
  ├── compression_ratio: "60.0"                        # 压缩比
  └── timestamp: "1234567890.123"                      # 压缩时间戳
TTL: 86400秒 (24小时)

# Set：会话摘要索引
qa:session:{session_id}:summary_keys
  成员: ["qa:summary:session_xxx:abc123", ...]

# Sorted Set：时间线（按时间排序）
qa:session:{session_id}:timeline
  成员: "qa:summary:session_xxx:abc123" (score=timestamp)
```

### 4.2 输出2：压缩过程解析文件

生成路径：`context_compact/analysis/compact_{session_id}_{summary_id}_YYYYMMDD_HHMMSS.json`

```json
{
    "分析文件信息": {
        "生成时间": "2026-07-15 10:30:00",
        "会话ID": "session_xxx",
        "摘要ID": "abc123",
        "压缩引擎版本": "1.0.0"
    },
    "一、压缩触发条件": {
        "触发机制": "当会话轮数超过阈值(5轮)时自动触发",
        "当前会话轮数": "6 轮",
        "压缩阈值": "5 轮",
        "是否达到阈值": true
    },
    "二、压缩前数据": {
        "原始文本长度": "500 字符",
        "原始文本预览": "[用户]: ...",
        "涉及商品/实体": ["HAK 180 烫金机"]
    },
    "三、压缩过程": {
        "步骤1": "从 MongoDB 获取会话历史记录",
        "步骤2": "格式化历史为[用户]/[助手]交替文本",
        "步骤3": "调用LLM(通义千问)进行语义压缩",
        "步骤4": "使用提示词模板: prompts/context_compact.prompt",
        "步骤5": "压缩策略: 保留核心Q&A + 去除冗余 + 保持实体完整性",
        "步骤6": "提取核心问题摘要和答案摘要"
    },
    "四、压缩结果": {
        "压缩后文本长度": "200 字符",
        "压缩比": "60%",
        "核心问题摘要": "HAK 180 烫金机操作指南",
        "核心答案摘要预览": "HAK 180 烫金机的操作步骤包括..."
    },
    "五、缓存策略": {
        "缓存目标": "Redis Hash (qa:summary:{session_id}:{summary_id})",
        "缓存TTL": "86400 秒 (24小时)"
    },
    "六、压缩评估": {
        "是否保留核心信息": true,
        "是否去除冗余": true
    }
}
```

### 4.3 使用方式

```python
# 1. 获取缓存管理器（单例）
from app.compression.cache_manager import get_cache_manager
cache_mgr = get_cache_manager()

# 2. 获取压缩引擎（单例）
from app.compression.context_compact_engine import get_compact_engine
engine = get_compact_engine()

# 3. 触发压缩（自动判断轮数）
result = engine.compact(session_id="xxx", history=history_list)
if result:
    summary_id = cache_mgr.save_summary("xxx", result)
    # 分析文件自动生成到 context_compact/analysis/

# 4. 检索缓存
summaries = cache_mgr.get_summaries_by_session("xxx")
from app.compression.relevance_judger import check_cache_relevance
hits = check_cache_relevance(user_query, summaries)
```

---

## 5. 三层缓存的集成关系

```
┌──────────────────────────────────────────────────────────────┐
│                    用户提问                                   │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  node_cache_check（缓存检查节点 - 本技能核心）                  │
│                                                                │
│  ┌──────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │ 步骤1    │ →  │ 步骤2            │ →  │ 步骤3          │  │
│  │ 读Redis  │    │ 相关性判断        │    │ 触发/缓存压缩   │  │
│  │ 缓存     │    │ (三级递进)        │    │ (≥5轮时)       │  │
│  └──────────┘    └────────┬─────────┘    └────────────────┘  │
│                           │                                    │
│                  ┌────────┴────────┐                          │
│                  ▼                  ▼                          │
│           ┌────────────┐   ┌──────────────┐                   │
│           │ 缓存命中    │   │ 缓存未命中    │                   │
│           │ score≥0.85 │   │ score<0.85   │                   │
│           │ →直出答案   │   │ →正常检索    │                   │
│           └────────────┘   └──────────────┘                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 依赖与配置

### 6.1 环境变量（.env）

```ini
# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_CACHE_TTL=86400
CACHE_RELEVANCE_THRESHOLD=0.85
COMPACT_TURN_THRESHOLD=5
```

### 6.2 关键模块

| 模块 | 路径 | 功能 |
|------|------|------|
| Redis配置 | `app/conf/redis_config.py` | Redis连接参数dataclass |
| Redis客户端 | `app/clients/redis_utils.py` | Redis单例连接管理/基础CRUD |
| 相关性判断 | `app/compression/relevance_judger.py` | 三级递进式语义相关性判断 |
| 压缩引擎 | `app/compression/context_compact_engine.py` | 对话历史语义压缩 |
| 缓存管理 | `app/compression/cache_manager.py` | Redis缓存读写/索引管理 |
| 缓存检查节点 | `app/query_process/agent/nodes/node_cache_check.py` | LangGraph缓存检查节点 |
| 压缩提示词 | `prompts/context_compact.prompt` | 压缩用的LLM提示词模板 |
| 相关性提示词 | `prompts/cache_relevance_check.prompt` | 相关性判断用的LLM提示词模板 |
| 压缩分析目录 | `context_compact/analysis/` | 压缩过程解析文件输出目录 |

### 6.3 方法引用

- **Cross-Encoder 相关性评分**：Nogueira & Cho, "Passage Re-ranking with BERT" (2019) https://arxiv.org/abs/1901.04085
- **BGE-Reranker**：Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding" (SIGIR 2023) https://arxiv.org/abs/2309.07597
- **Bi-Encoder 语义相似度**：Reimers & Gurevych, "Sentence-BERT" (EMNLP 2019) https://arxiv.org/abs/1908.10084
- **LLM-as-Judge**：Zheng et al., "Judging LLM-as-a-Judge" (NeurIPS 2023) https://arxiv.org/abs/2306.05685
