"""
Ragas 评估框架 - 配置模块
定义评估相关的阈值、路径、模型等配置
"""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EvalConfig:
    """Ragas 评估配置"""
    # ==================== 评估阈值 ====================
    # ContextPrecision 最低阈值（低于此值触发重试/中断）
    context_precision_threshold: float = 0.7
    # ContextRecall 最低阈值
    context_recall_threshold: float = 0.6
    # ResponseRelevancy 最低阈值
    response_relevancy_threshold: float = 0.7
    # Faithfulness 最低阈值
    faithfulness_threshold: float = 0.8

    # ==================== 中断机制 ====================
    # 最大评估失败次数（超过此值触发中断）
    max_eval_fail_count: int = 2
    # 评估失败计数器名称
    eval_fail_counter_key: str = "eval_fail_count"

    # ==================== 路径配置 ====================
    # 测试数据集路径
    test_dataset_path: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "doc", "eval", "test_dataset.json"
    )

    # ==================== 评估模型 ====================
    # Ragas 评估时使用的 LLM（复用项目已有的 qwen-flash）
    eval_llm_model: str = "qwen-flash"
    # Ragas 评估时使用的 Embedding（复用项目已有的 BGE-M3）
    eval_embedding_model: str = "BAAI/bge-m3"


# 全局配置实例
eval_config = EvalConfig()
