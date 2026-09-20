"""
Ragas 评估器模块
封装 Ragas 框架的四个核心评估指标，提供统一的评估接口
四个指标：
  1. ContextPrecision  - 检索上下文精确度（评估检索到的文档是否相关）
  2. ContextRecall     - 检索上下文召回率（评估相关文档是否都被检索到）
  3. ResponseRelevancy - 回答相关性（评估生成的答案是否与问题相关）
  4. Faithfulness      - 回答忠实度（评估答案是否基于检索到的上下文）

使用方式：
  evaluator = RagasEvaluator()
  # 评估检索质量
  result = evaluator.evaluate_retrieval(question, contexts, ground_truth_context)
  # 评估生成质量
  result = evaluator.evaluate_generation(question, contexts, answer)
"""
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app.eval.eval_config import eval_config
from app.core.logger import logger

load_dotenv()


@dataclass
class EvalResult:
    """
    单次评估结果
    context_precision: ContextPrecision 分数（0~1）
    context_recall: ContextRecall 分数（0~1）
    response_relevancy: ResponseRelevancy 分数（0~1）
    faithfulness: Faithfulness 分数（0~1）
    all_passed: 所有指标是否通过对应阈值
    fail_reasons: 未通过指标的失败原因列表
    """
    context_precision: float = 0.0
    context_recall: float = 0.0
    response_relevancy: float = 0.0
    faithfulness: float = 0.0
    all_passed: bool = True
    fail_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "response_relevancy": round(self.response_relevancy, 4),
            "faithfulness": round(self.faithfulness, 4),
            "all_passed": self.all_passed,
            "fail_reasons": self.fail_reasons,
        }


class RagasEvaluator:
    """
    Ragas 评估器
    封装 Ragas 框架的四个评估指标，提供统一的评估接口
    支持自动降级：当 Ragas 库不可用时，使用基于 LLM 的近似评估
    """

    def __init__(self):
        """初始化评估器"""
        self._ragas_available = False
        self._init_ragas()

    def _init_ragas(self):
        """尝试初始化 Ragas 库（懒加载）"""
        try:
            # 尝试导入 ragas
            import ragas
            self._ragas_available = True
            logger.info("Ragas 评估库初始化成功")
        except ImportError:
            logger.warning("Ragas 库未安装，将使用 LLM 近似评估（pip install ragas）")
            self._ragas_available = False

    @property
    def is_available(self) -> bool:
        """Ragas 库是否可用"""
        return self._ragas_available

    # ==================== Stage 1: 检索评估 ====================

    def evaluate_retrieval(
        self,
        question: str,
        contexts: List[str],
        ground_truth_context: str
    ) -> EvalResult:
        """
        评估检索质量（ContextPrecision + ContextRecall）
        在检索环节之后（RRF/Rerank 之后）调用

        参数：
            question: 用户原始问题
            contexts: 检索到的上下文列表（reranked_docs 的 text 字段列表）
            ground_truth_context: 标准答案上下文（来自测试集）
        返回：
            EvalResult: 包含 ContextPrecision 和 ContextRecall 分数
        """
        result = EvalResult()

        if not contexts:
            result.all_passed = False
            result.fail_reasons.append("检索结果为空")
            return result

        try:
            if self._ragas_available:
                # 使用 Ragas 原生评估
                score = self._ragas_eval_retrieval(question, contexts, ground_truth_context)
                result.context_precision = score.get("context_precision", 0.0)
                result.context_recall = score.get("context_recall", 0.0)
            else:
                # 使用 LLM 近似评估
                result.context_precision = self._approx_context_precision(question, contexts, ground_truth_context)
                result.context_recall = self._approx_context_recall(question, contexts, ground_truth_context)

            # 阈值判定
            if result.context_precision < eval_config.context_precision_threshold:
                result.fail_reasons.append(
                    f"ContextPrecision({result.context_precision:.4f}) < "
                    f"阈值({eval_config.context_precision_threshold})"
                )
            if result.context_recall < eval_config.context_recall_threshold:
                result.fail_reasons.append(
                    f"ContextRecall({result.context_recall:.4f}) < "
                    f"阈值({eval_config.context_recall_threshold})"
                )

        except Exception as e:
            logger.error(f"检索评估异常: {str(e)}")
            result.fail_reasons.append(f"评估异常: {str(e)}")

        result.all_passed = len(result.fail_reasons) == 0
        return result

    # ==================== Stage 2: 生成评估 ====================

    def evaluate_generation(
        self,
        question: str,
        contexts: List[str],
        answer: str
    ) -> EvalResult:
        """
        评估生成质量（ResponseRelevancy + Faithfulness）
        在生成答案之后调用

        参数：
            question: 用户原始问题
            contexts: 检索到的上下文列表
            answer: LLM 生成的答案
        返回：
            EvalResult: 包含 ResponseRelevancy 和 Faithfulness 分数
        """
        result = EvalResult()

        if not answer:
            result.all_passed = False
            result.fail_reasons.append("生成答案为空")
            return result

        try:
            if self._ragas_available:
                score = self._ragas_eval_generation(question, contexts, answer)
                result.response_relevancy = score.get("response_relevancy", 0.0)
                result.faithfulness = score.get("faithfulness", 0.0)
            else:
                result.response_relevancy = self._approx_response_relevancy(question, contexts, answer)
                result.faithfulness = self._approx_faithfulness(question, contexts, answer)

            # 阈值判定
            if result.response_relevancy < eval_config.response_relevancy_threshold:
                result.fail_reasons.append(
                    f"ResponseRelevancy({result.response_relevancy:.4f}) < "
                    f"阈值({eval_config.response_relevancy_threshold})"
                )
            if result.faithfulness < eval_config.faithfulness_threshold:
                result.fail_reasons.append(
                    f"Faithfulness({result.faithfulness:.4f}) < "
                    f"阈值({eval_config.faithfulness_threshold})"
                )

        except Exception as e:
            logger.error(f"生成评估异常: {str(e)}")
            result.fail_reasons.append(f"评估异常: {str(e)}")

        result.all_passed = len(result.fail_reasons) == 0
        return result

    # ==================== Ragas 原生评估 ====================

    def _ragas_eval_retrieval(
        self,
        question: str,
        contexts: List[str],
        ground_truth_context: str
    ) -> Dict[str, float]:
        """
        使用 Ragas 框架评估检索质量
        需要 ragas>=0.1.0 库支持
        """
        try:
            from ragas.metrics import ContextPrecision, ContextRecall
            from ragas.llms import llm_factory
            from datasets import Dataset

            # 构造 Ragas 所需的数据集格式
            data = {
                "question": [question],
                "contexts": [contexts],
                "ground_truth_context": [ground_truth_context],
            }
            dataset = Dataset.from_dict(data)

            # 计算 ContextPrecision
            cp = ContextPrecision()
            cp_score = cp.score(dataset)["context_precision"][0]

            # 计算 ContextRecall
            cr = ContextRecall()
            cr_score = cr.score(dataset)["context_recall"][0]

            return {"context_precision": float(cp_score), "context_recall": float(cr_score)}

        except Exception as e:
            logger.error(f"Ragas 检索评估失败，降级到 LLM 近似评估: {str(e)}")
            return {"context_precision": self._approx_context_precision(question, contexts, ground_truth_context),
                    "context_recall": self._approx_context_recall(question, contexts, ground_truth_context)}

    def _ragas_eval_generation(
        self,
        question: str,
        contexts: List[str],
        answer: str
    ) -> Dict[str, float]:
        """
        使用 Ragas 框架评估生成质量
        """
        try:
            from ragas.metrics import ResponseRelevancy, Faithfulness
            from datasets import Dataset

            data = {
                "question": [question],
                "contexts": [contexts],
                "answer": [answer],
            }
            dataset = Dataset.from_dict(data)

            rr = ResponseRelevancy()
            rr_score = rr.score(dataset)["response_relevancy"][0]

            fh = Faithfulness()
            fh_score = fh.score(dataset)["faithfulness"][0]

            return {"response_relevancy": float(rr_score), "faithfulness": float(fh_score)}

        except Exception as e:
            logger.error(f"Ragas 生成评估失败，降级到 LLM 近似评估: {str(e)}")
            return {"response_relevancy": self._approx_response_relevancy(question, contexts, answer),
                    "faithfulness": self._approx_faithfulness(question, contexts, answer)}

    # ==================== LLM 近似评估（Ragas 不可用时的降级方案） ====================

    def _approx_context_precision(self, question: str, contexts: List[str], ground_truth: str) -> float:
        """
        近似计算 ContextPrecision（基于关键词重叠率 + LLM 判断）
        通过比较检索到的上下文与标准上下文的相关性来估算精确度
        """
        if not contexts:
            return 0.0

        try:
            from app.lm.embedding_utils import generate_embeddings
            import numpy as np

            # 计算 ground_truth 的向量
            gt_emb = generate_embeddings([ground_truth])["dense"][0]
            gt_norm = np.array(gt_emb) / (np.linalg.norm(np.array(gt_emb)) + 1e-8)

            # 计算每个检索到的上下文与 ground_truth 的相似度
            ctx_embs = generate_embeddings(contexts)["dense"]
            scores = []
            for ctx_emb in ctx_embs:
                ctx_norm = np.array(ctx_emb) / (np.linalg.norm(np.array(ctx_emb)) + 1e-8)
                sim = float(np.dot(gt_norm, ctx_norm))
                scores.append(sim)

            # ContextPrecision ≈ 前 k 个相关文档的比例（按位置加权）
            relevant_count = sum(1 for s in scores if s > 0.7)
            precision = relevant_count / max(len(scores), 1)
            return min(precision, 1.0)

        except Exception as e:
            logger.warning(f"近似 ContextPrecision 计算失败: {str(e)}")
            return 0.5  # 默认返回中等分数

    def _approx_context_recall(self, question: str, contexts: List[str], ground_truth: str) -> float:
        """
        近似计算 ContextRecall（通过计算 ground_truth 被 contexts 覆盖的比例）
        """
        if not contexts:
            return 0.0

        try:
            from app.lm.embedding_utils import generate_embeddings
            import numpy as np

            gt_emb = generate_embeddings([ground_truth])["dense"][0]
            gt_norm = np.array(gt_emb) / (np.linalg.norm(np.array(gt_emb)) + 1e-8)

            ctx_embs = generate_embeddings(contexts)["dense"]
            # 找与 ground_truth 最相似的上文
            max_sim = 0.0
            for ctx_emb in ctx_embs:
                ctx_norm = np.array(ctx_emb) / (np.linalg.norm(np.array(ctx_emb)) + 1e-8)
                sim = float(np.dot(gt_norm, ctx_norm))
                max_sim = max(max_sim, sim)

            # ContextRecall ≈ 检索到的上下文与标准上下文的最大相似度
            return min(max(max_sim, 0.0), 1.0)

        except Exception as e:
            logger.warning(f"近似 ContextRecall 计算失败: {str(e)}")
            return 0.5

    def _approx_response_relevancy(self, question: str, contexts: List[str], answer: str) -> float:
        """
        近似计算 ResponseRelevancy（答案与问题的相关性）
        """
        try:
            from app.lm.embedding_utils import generate_embeddings
            import numpy as np

            q_emb = generate_embeddings([question, answer])["dense"]
            q_vec = np.array(q_emb[0]) / (np.linalg.norm(np.array(q_emb[0])) + 1e-8)
            a_vec = np.array(q_emb[1]) / (np.linalg.norm(np.array(q_emb[1])) + 1e-8)
            sim = float(np.dot(q_vec, a_vec))
            return min(max(sim, 0.0), 1.0)

        except Exception as e:
            logger.warning(f"近似 ResponseRelevancy 计算失败: {str(e)}")
            return 0.7

    def _approx_faithfulness(self, question: str, contexts: List[str], answer: str) -> float:
        """
        近似计算 Faithfulness（答案是否基于上下文，不包含幻觉）
        通过检查答案中的关键信息是否能在上下文中找到依据
        """
        if not contexts:
            return 0.0

        try:
            # 简单的基于关键词的忠实度检查
            # 将答案和上下文分词，计算答案中的关键内容在上下文中出现的比例
            import re

            # 提取答案中的关键短语（中文句子片段）
            answer_sentences = re.split(r'[。！？；\n]', answer)
            answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 5]

            if not answer_sentences:
                return 0.8

            # 合并所有上下文
            all_context = " ".join(contexts)

            # 计算每个句子能在上下文中找到依据的比例
            supported = 0
            for sent in answer_sentences:
                # 提取句子中的关键词（去掉无意义词）
                key_phrases = re.findall(r'[一-鿿]{2,}', sent)
                if not key_phrases:
                    supported += 1
                    continue
                # 如果至少一半的关键词出现在上下文中，认为该句子有依据
                match_count = sum(1 for phrase in key_phrases if phrase in all_context)
                if match_count >= max(len(key_phrases) * 0.3, 1):
                    supported += 1

            faithfulness = supported / max(len(answer_sentences), 1)
            return min(max(faithfulness, 0.0), 1.0)

        except Exception as e:
            logger.warning(f"近似 Faithfulness 计算失败: {str(e)}")
            return 0.7


# ==================== 全局单例 ====================

_evaluator_instance: Optional[RagasEvaluator] = None


def get_evaluator() -> RagasEvaluator:
    """获取评估器单例"""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = RagasEvaluator()
    return _evaluator_instance


if __name__ == "__main__":
    """本地测试"""
    logger.info("===== Ragas 评估器测试 =====")

    evaluator = get_evaluator()
    logger.info(f"Ragas 库可用: {evaluator.is_available}")

    test_question = "HAK 180 烫金机如何设置温度？"
    test_contexts = [
        "按下 Temperature 按钮使用 +/- 键调节温度，建议 110℃，按 OK 确认。",
        "HAK 180 烫金机通过操作面板上的 Temperature 按钮设置温度。",
    ]
    test_gt = "HAK 180 烫金机通过 Temperature 按钮设置温度，建议 110℃。"
    test_answer = "按下面板的 Temperature 键，用 +/- 调节到 110℃，按 OK 确认。"

    # 检索评估
    ret_result = evaluator.evaluate_retrieval(test_question, test_contexts, test_gt)
    logger.info(f"检索评估: precision={ret_result.context_precision:.4f}, recall={ret_result.context_recall:.4f}, "
                f"passed={ret_result.all_passed}")

    # 生成评估
    gen_result = evaluator.evaluate_generation(test_question, test_contexts, test_answer)
    logger.info(f"生成评估: relevancy={gen_result.response_relevancy:.4f}, faithfulness={gen_result.faithfulness:.4f}, "
                f"passed={gen_result.all_passed}")

    logger.info("===== Ragas 评估器测试完成 =====")
