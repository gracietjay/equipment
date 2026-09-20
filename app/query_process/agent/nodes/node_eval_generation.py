"""
生成评估节点 - ResponseRelevancy + Faithfulness
位置：在生成答案之后、最终输出之前
功能：
  1. 获取已生成的答案和检索到的上下文
  2. 使用 Ragas 框架评估 ResponseRelevancy（答案与问题的相关性）
     和 Faithfulness（答案对上下文的忠实度）
  3. 如果评估不达标，可触发重生成或中断等待人工审核
  4. 中断后可经人工审核后恢复工作流

与 node_eval_retrieval 的区别：
  - node_eval_retrieval：评估检索质量（ContextPrecision + ContextRecall）
  - node_eval_generation：评估生成质量（ResponseRelevancy + Faithfulness）
"""
import sys
from typing import Dict, Any, List

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.eval.ragas_evaluator import get_evaluator, RagasEvaluator, EvalResult
from app.eval.interrupt_handler import get_interrupt_handler, InterruptHandler


def step_1_prepare_data(state: QueryGraphState) -> tuple:
    """
    步骤一：准备评估所需的数据
    参数：
        state: 当前查询状态
    返回：
        (question, contexts, answer) 三元组
    """
    question = state.get("rewritten_query") or state.get("original_query", "")
    answer = state.get("answer", "")

    # 从 reranked_docs 提取上下文
    reranked_docs = state.get("reranked_docs") or []
    contexts = []
    for doc in reranked_docs:
        text = doc.get("text") or doc.get("content") or ""
        if text.strip():
            contexts.append(text.strip())

    logger.info(f"[生成评估] 步骤一: 数据就绪 - question_len={len(question)}, "
                f"answer_len={len(answer)}, contexts={len(contexts)}条")

    return question, contexts, answer


def step_2_check_cache_source(state: QueryGraphState) -> bool:
    """
    步骤二：检查答案来源
    如果答案来自 Redis 缓存（from_cache=True），跳过 Faithfulness 评估
    因为缓存答案是基于之前已验证过的上下文生成的
    参数：
        state: 当前查询状态
    返回：
        True=需要评估, False=跳过评估
    """
    if state.get("from_cache"):
        logger.info("[生成评估] 步骤二: 答案来自 Redis 缓存，跳过生成评估")
        return False
    return True


def step_3_run_evaluation(
    evaluator: RagasEvaluator,
    question: str,
    contexts: List[str],
    answer: str
) -> EvalResult:
    """
    步骤三：执行 Ragas 生成评估
    评估 ResponseRelevancy 和 Faithfulness
    参数：
        evaluator: Ragas 评估器实例
        question: 用户提问
        contexts: 检索到的上下文
        answer: 生成的答案
    返回：
        EvalResult 评估结果
    """
    logger.info(f"[生成评估] 步骤三: 执行评估 - question={question[:30]}..., answer_len={len(answer)}")

    result = evaluator.evaluate_generation(question, contexts, answer)

    logger.info(f"[生成评估] 步骤三: 评估完成 - "
                f"ResponseRelevancy={result.response_relevancy:.4f}, "
                f"Faithfulness={result.faithfulness:.4f}, "
                f"passed={result.all_passed}")
    return result


def step_4_handle_failure(
    handler: InterruptHandler,
    session_id: str,
    eval_result: EvalResult
) -> bool:
    """
    步骤四：处理生成评估不达标的情况
    参数：
        handler: 中断处理器实例
        session_id: 会话 ID
        eval_result: 评估结果
    返回：
        True=触发了中断, False=未触发
    """
    if eval_result.all_passed:
        handler.reset_fail_count(session_id)
        logger.info(f"[生成评估] ✅ 会话 {session_id} 生成评估通过")
        return False

    triggered = handler.check_and_trigger(
        session_id=session_id,
        eval_stage="generation",
        eval_result=eval_result.to_dict(),
        fail_reasons=eval_result.fail_reasons,
    )

    if triggered:
        logger.warning(f"[生成评估] ⚠️ 会话 {session_id} 生成评估触发中断，等待人工审核")
    else:
        logger.info(f"[生成评估] 会话 {session_id} 生成评估未达标（可重试）")

    return triggered


def node_eval_generation(state: QueryGraphState) -> QueryGraphState:
    """
    生成评估节点（ResponseRelevancy + Faithfulness）
    在答案生成之后、最终 SSE 推送之前执行
    工作流位置：node_answer_output → node_eval_generation → (通过→输出 / 中断→审核)

    返回状态更新：
      - eval_generation_relevancy: ResponseRelevancy 分数
      - eval_generation_faithfulness: Faithfulness 分数
      - eval_generation_passed: 评估是否通过
      - needs_interrupt: 是否需要中断
      - interrupt_stage: 中断阶段

    注意：
      - 如果答案来自缓存（from_cache=True），跳过生成评估
      - 评估失败时通过中断处理器决定是否触发人工审核
    """
    logger.info("--- node_eval_generation (生成评估) 节点开始处理 ---")
    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)

    add_running_task(session_id, sys._getframe().f_code.co_name, is_stream)

    # 初始化组件
    evaluator = get_evaluator()
    handler = get_interrupt_handler()

    # 步骤一：准备数据
    question, contexts, answer = step_1_prepare_data(state)

    if not answer:
        logger.warning("[生成评估] 答案为空，跳过生成评估")
        state["eval_generation_passed"] = True
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
        return state

    # 步骤二：检查答案来源（缓存答案跳过评估）
    if not step_2_check_cache_source(state):
        state["eval_generation_passed"] = True
        add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
        return state

    # 步骤三：执行评估
    eval_result = step_3_run_evaluation(evaluator, question, contexts, answer)

    # 更新 state
    state["eval_generation_relevancy"] = eval_result.response_relevancy
    state["eval_generation_faithfulness"] = eval_result.faithfulness
    state["eval_generation_passed"] = eval_result.all_passed

    # 步骤四：处理失败
    if not eval_result.all_passed:
        triggered = step_4_handle_failure(handler, session_id, eval_result)
        if triggered:
            state["needs_interrupt"] = True
            state["interrupt_stage"] = "generation"
            interrupt = handler.get_interrupt(session_id)
            if interrupt:
                state["interrupt_id"] = interrupt.get("interrupt_id", "")
            logger.warning(f"[生成评估] ⚠️ 生成评估中断已触发，等待人工审核")
    else:
        logger.info(f"[生成评估] ✅ 生成评估全部通过")

    add_done_task(session_id, sys._getframe().f_code.co_name, is_stream)
    logger.info("--- node_eval_generation 节点处理结束 ---")

    return state


if __name__ == "__main__":
    """本地测试"""
    logger.info("===== 生成评估节点测试 =====")

    test_state = {
        "session_id": "test_eval_gen",
        "original_query": "HAK 180 烫金机如何设置温度？",
        "rewritten_query": "HAK 180 烫金机温度设置方法",
        "answer": "按下面板的 Temperature 键，用 +/- 调节到 110℃，按 OK 确认。",
        "reranked_docs": [
            {"text": "按下 Temperature 按钮使用 +/- 键调节温度，建议 110℃，按 OK 确认。"},
        ],
        "from_cache": False,
        "is_stream": False,
    }

    result = node_eval_generation(test_state)
    logger.info(f"评估结果: relevancy={result.get('eval_generation_relevancy')}, "
                f"faithfulness={result.get('eval_generation_faithfulness')}, "
                f"passed={result.get('eval_generation_passed')}")
    logger.info("===== 生成评估节点测试完成 =====")
