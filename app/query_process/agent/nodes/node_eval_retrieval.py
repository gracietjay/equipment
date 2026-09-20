"""
检索评估节点 - ContextPrecision + ContextRecall
位置：在 Rerank 节点之后，回答生成之前
功能：
  1. 从 state 中获取检索结果（reranked_docs）和预期上下文
  2. 使用 Ragas 框架评估 ContextPrecision 和 ContextRecall
  3. 如果评估不达标，递增失败计数
  4. 当连续失败超过阈值时，触发中断等待人工审核
  5. 中断后可经人工审核后恢复工作流

中断机制：
  - 无人值守模式：评估不达标时自动重试（最多重试 max_eval_fail_count 次）
  - 人工介入模式：重试耗尽后触发中断，等待 /eval/review 接口审核
  - 审核后恢复：人工审核通过（approved）后重置计数继续执行
"""
import sys
import json
from typing import Dict, Any, List, Optional

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.eval.eval_config import eval_config
from app.eval.ragas_evaluator import get_evaluator, RagasEvaluator, EvalResult
from app.eval.interrupt_handler import get_interrupt_handler, InterruptHandler


def step_1_prepare_contexts(state: QueryGraphState) -> List[str]:
    """
    步骤一：从检索结果中提取上下文文本列表
    参数：
        state: 当前查询状态
    返回：
        上下文字符串列表
    """
    reranked_docs = state.get("reranked_docs") or []
    contexts = []

    for doc in reranked_docs:
        text = doc.get("text") or doc.get("content") or ""
        if text.strip():
            contexts.append(text.strip())

    logger.info(f"[检索评估] 步骤一: 从 reranked_docs 提取到 {len(contexts)} 条上下文")
    return contexts


def step_2_load_ground_truth(state: QueryGraphState) -> Optional[str]:
    """
    步骤二：加载标准答案上下文
    优先从 state 中读取 eval_ground_truth，其次尝试从测试集文件加载
    参数：
        state: 当前查询状态
    返回：
        标准上下文文本
    """
    # 优先使用 state 中预设的标准上下文
    ground_truth = state.get("eval_ground_truth")
    if ground_truth:
        logger.debug(f"[检索评估] 步骤二: 使用 state 中的标准上下文 (长度 {len(ground_truth)})")
        return ground_truth

    # 尝试从测试集加载
    try:
        import os
        dataset_path = eval_config.test_dataset_path
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            # 尝试匹配问题
            question = state.get("original_query", "")
            for entry in dataset:
                if entry.get("question", "") in question or question in entry.get("question", ""):
                    ground_truth = entry.get("ground_truth_context", "")
                    logger.info(f"[检索评估] 步骤二: 从测试集匹配到标准上下文 (问题: {entry['question'][:30]}...)")
                    return ground_truth

            # 未匹配到测试集条目，使用第一条的 ground_truth 作为兜底
            if dataset:
                ground_truth = dataset[0].get("ground_truth_context", "")
                logger.warning(f"[检索评估] 步骤二: 未匹配到测试集条目，使用第一条作为兜底")
                return ground_truth
    except Exception as e:
        logger.warning(f"[检索评估] 步骤二: 加载测试集失败: {str(e)}")

    logger.warning("[检索评估] 步骤二: 未找到标准上下文，无法评估")
    return None


def step_3_run_evaluation(
    evaluator: RagasEvaluator,
    question: str,
    contexts: List[str],
    ground_truth: str
) -> EvalResult:
    """
    步骤三：执行 Ragas 检索评估
    评估 ContextPrecision 和 ContextRecall
    参数：
        evaluator: Ragas 评估器实例
        question: 用户提问
        contexts: 检索到的上下文列表
        ground_truth: 标准上下文
    返回：
        EvalResult 评估结果
    """
    logger.info(f"[检索评估] 步骤三: 执行评估（question={question[:30]}..., "
                f"contexts={len(contexts)}条, gt_len={len(ground_truth)}）")

    result = evaluator.evaluate_retrieval(question, contexts, ground_truth)

    logger.info(f"[检索评估] 步骤三: 评估完成 - "
                f"ContextPrecision={result.context_precision:.4f}, "
                f"ContextRecall={result.context_recall:.4f}, "
                f"passed={result.all_passed}")
    return result


def step_4_handle_failure(
    handler: InterruptHandler,
    session_id: str,
    eval_result: EvalResult
) -> bool:
    """
    步骤四：处理评估不达标的情况
    如果连续失败超过阈值，触发中断
    参数：
        handler: 中断处理器实例
        session_id: 会话 ID
        eval_result: 评估结果
    返回：
        True=触发了中断，False=尚未触发（可继续重试）
    """
    if eval_result.all_passed:
        # 评估通过，重置计数
        handler.reset_fail_count(session_id)
        logger.info(f"[检索评估] ✅ 会话 {session_id} 检索评估通过")
        return False

    # 评估不达标，检查是否触发中断
    triggered = handler.check_and_trigger(
        session_id=session_id,
        eval_stage="retrieval",
        eval_result=eval_result.to_dict(),
        fail_reasons=eval_result.fail_reasons,
        state_snapshot={
            "session_id": session_id,
            "question": "（见 state.original_query）",
        }
    )

    if triggered:
        logger.warning(f"[检索评估] ⚠️ 会话 {session_id} 检索评估触发中断，等待人工审核")
    else:
        logger.info(f"[检索评估] 会话 {session_id} 检索评估未达标（可重试）")

    return triggered


def node_eval_retrieval(state: QueryGraphState) -> QueryGraphState:
    """
    检索评估节点（ContextPrecision + ContextRecall）
    在 Rerank 之后、回答生成之前执行
    工作流位置：node_rerank → node_eval_retrieval → (通过→node_answer_output / 中断)

    返回状态更新：
      - eval_retrieval_precision: ContextPrecision 分数
      - eval_retrieval_recall: ContextRecall 分数
      - eval_retrieval_passed: 评估是否通过
      - eval_retrieval_fail_count: 连续失败次数
      - needs_interrupt: 是否需要中断
      - interrupt_stage: 中断阶段
    """
    logger.info("--- node_eval_retrieval (检索评估) 节点开始处理 ---")
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    # 初始化组件
    evaluator = get_evaluator()
    handler = get_interrupt_handler()

    # 步骤一：准备上下文
    contexts = step_1_prepare_contexts(state)
    if not contexts:
        logger.warning("[检索评估] 检索结果为空，跳过评估")
        state["eval_retrieval_passed"] = True  # 跳过评估
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
        return state

    # 步骤二：加载标准上下文
    ground_truth = step_2_load_ground_truth(state)
    if not ground_truth:
        logger.warning("[检索评估] 无标准上下文，跳过评估")
        state["eval_retrieval_passed"] = True
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
        return state

    # 步骤三：执行评估
    question = state.get("rewritten_query") or state.get("original_query", "")
    eval_result = step_3_run_evaluation(evaluator, question, contexts, ground_truth)

    # 更新 state
    state["eval_retrieval_precision"] = eval_result.context_precision
    state["eval_retrieval_recall"] = eval_result.context_recall
    state["eval_retrieval_passed"] = eval_result.all_passed
    state["eval_retrieval_fail_count"] = handler.get_fail_count(session_id)

    # 步骤四：处理失败
    if not eval_result.all_passed:
        triggered = step_4_handle_failure(handler, session_id, eval_result)
        if triggered:
            state["needs_interrupt"] = True
            state["interrupt_stage"] = "retrieval"
            # 获取中断 ID
            interrupt = handler.get_interrupt(session_id)
            if interrupt:
                state["interrupt_id"] = interrupt.get("interrupt_id", "")
            logger.warning(f"[检索评估] ⚠️ 检索评估中断已触发，等待人工审核")
    else:
        # 评估通过
        logger.info(f"[检索评估] ✅ 检索评估全部通过")

    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    logger.info("--- node_eval_retrieval 节点处理结束 ---")

    return state


if __name__ == "__main__":
    """本地测试"""
    logger.info("===== 检索评估节点测试 =====")

    test_state = {
        "session_id": "test_eval_retrieval",
        "original_query": "HAK 180 烫金机如何设置温度？",
        "rewritten_query": "HAK 180 烫金机温度设置方法",
        "reranked_docs": [
            {"text": "按下 Temperature 按钮使用 +/- 键调节温度，建议 110℃，按 OK 确认。"},
            {"text": "HAK 180 烫金机通过操作面板上的 Temperature 按钮设置温度。"},
        ],
        "eval_ground_truth": "HAK 180 烫金机通过 Temperature 按钮设置温度，建议 110℃。",
        "is_stream": False,
    }

    result = node_eval_retrieval(test_state)
    logger.info(f"评估结果: precision={result.get('eval_retrieval_precision')}, "
                f"recall={result.get('eval_retrieval_recall')}, "
                f"passed={result.get('eval_retrieval_passed')}")
    logger.info("===== 检索评估节点测试完成 =====")
