"""
中断处理器模块
功能：管理评估不达标时的中断（Interrupt）机制，支持人工审核后恢复工作流
核心逻辑：
  1. 当评估指标连续 N 次低于阈值时，触发中断
  2. 中断后暂停工作流，等待人工审核
  3. 人工审核后可选择：批准（继续）/ 拒绝（重试）/ 修改（调参后重试）
  4. 中断状态持久化到内存字典，通过 API 接口与前端交互

设计原则：
  - 无人值守模式：评估不达标时自动重试（最多重试 max_retry 次）
  - 人工介入模式：重试耗尽后暂停，等待人工审核
  - 审核后恢复：保留已检索的上下文，支持在人工指导下优化
"""
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from app.core.logger import logger
from app.eval.eval_config import eval_config


@dataclass
class InterruptRecord:
    """
    中断记录
    interrupt_id: 中断唯一标识
    session_id: 关联的会话 ID
    eval_stage: 中断发生的评估阶段（retrieval / generation）
    fail_reasons: 导致中断的失败原因列表
    fail_count: 连续失败次数
    eval_result: 最后一次评估的详细结果
    state_snapshot: 中断时的状态快照（用于恢复）
    status: 中断状态（pending / approved / rejected / modified）
    created_at: 创建时间
    reviewed_at: 审核时间
    review_comment: 审核意见
    """
    interrupt_id: str = ""
    session_id: str = ""
    eval_stage: str = ""
    fail_reasons: List[str] = field(default_factory=list)
    fail_count: int = 0
    eval_result: Dict[str, Any] = field(default_factory=dict)
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending / approved / rejected / modified
    created_at: str = ""
    reviewed_at: str = ""
    review_comment: str = ""


class InterruptHandler:
    """
    中断处理器
    管理评估不达标时的中断/恢复全生命周期
    线程安全（单进程内存存储，与 task_utils 风格一致）
    """

    def __init__(self):
        # {session_id: InterruptRecord}
        self._interrupts: Dict[str, InterruptRecord] = {}
        # {session_id: fail_count} 连续失败计数
        self._fail_counts: Dict[str, int] = {}

    def check_and_trigger(
        self,
        session_id: str,
        eval_stage: str,
        eval_result: Dict[str, Any],
        fail_reasons: List[str],
        state_snapshot: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        检查评估结果，如果连续失败超过阈值则触发中断
        参数：
            session_id: 会话 ID
            eval_stage: 评估阶段（retrieval / generation）
            eval_result: 评估结果字典
            fail_reasons: 失败原因列表
            state_snapshot: 中断时的状态快照
        返回：
            True=已触发中断，False=未触发（可继续重试）
        """
        # 更新连续失败计数
        current_count = self._fail_counts.get(session_id, 0)
        if fail_reasons:
            current_count += 1
        else:
            # 评估通过，重置计数
            self._fail_counts[session_id] = 0
            return False

        self._fail_counts[session_id] = current_count
        logger.info(f"[中断] 会话 {session_id} {eval_stage} 评估第 {current_count} 次失败: {fail_reasons}")

        # 判断是否达到中断阈值
        if current_count >= eval_config.max_eval_fail_count:
            self._trigger_interrupt(session_id, eval_stage, eval_result, fail_reasons, state_snapshot)
            return True

        return False

    def _trigger_interrupt(
        self,
        session_id: str,
        eval_stage: str,
        eval_result: Dict[str, Any],
        fail_reasons: List[str],
        state_snapshot: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        触发中断，创建中断记录等待人工审核
        """
        record = InterruptRecord(
            interrupt_id=str(uuid.uuid4())[:8],
            session_id=session_id,
            eval_stage=eval_stage,
            fail_reasons=fail_reasons,
            fail_count=self._fail_counts.get(session_id, 0),
            eval_result=eval_result,
            state_snapshot=state_snapshot or {},
            status="pending",
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._interrupts[session_id] = record
        logger.warning(
            f"[中断] ⚠️ 会话 {session_id} 触发中断！"
            f"阶段={eval_stage}, 失败次数={record.fail_count}, "
            f"中断ID={record.interrupt_id}"
        )

    # ==================== 审核接口 ====================

    def get_pending_interrupts(self) -> List[Dict[str, Any]]:
        """获取所有待审核的中断列表"""
        pending = []
        for sid, record in self._interrupts.items():
            if record.status == "pending":
                pending.append({
                    "session_id": sid,
                    "interrupt_id": record.interrupt_id,
                    "eval_stage": record.eval_stage,
                    "fail_reasons": record.fail_reasons,
                    "fail_count": record.fail_count,
                    "eval_result": record.eval_result,
                    "created_at": record.created_at,
                })
        return pending

    def get_interrupt(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取指定会话的中断详情"""
        record = self._interrupts.get(session_id)
        if record is None:
            return None
        return {
            "session_id": session_id,
            "interrupt_id": record.interrupt_id,
            "eval_stage": record.eval_stage,
            "fail_reasons": record.fail_reasons,
            "fail_count": record.fail_count,
            "eval_result": record.eval_result,
            "status": record.status,
            "created_at": record.created_at,
            "reviewed_at": record.reviewed_at,
            "review_comment": record.review_comment,
        }

    def review_interrupt(
        self,
        session_id: str,
        action: str,
        comment: str = ""
    ) -> bool:
        """
        人工审核中断
        参数：
            session_id: 会话 ID
            action: 审核动作（approved=批准通过 / rejected=拒绝重试 / modified=调整后恢复）
            comment: 审核意见
        返回：
            True=审核成功，False=未找到中断记录
        """
        record = self._interrupts.get(session_id)
        if record is None:
            logger.warning(f"[中断] 未找到会话 {session_id} 的中断记录")
            return False

        if record.status != "pending":
            logger.warning(f"[中断] 会话 {session_id} 的中断已审核（status={record.status}）")
            return False

        record.status = action
        record.reviewed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record.review_comment = comment

        if action == "approved":
            # 批准：重置失败计数，允许继续
            self._fail_counts[session_id] = 0
            logger.info(f"[中断] ✅ 会话 {session_id} 已批准，工作流继续执行")
        elif action == "rejected":
            # 拒绝：标记为最终失败
            logger.info(f"[中断] ❌ 会话 {session_id} 已拒绝，工作流终止")
        elif action == "modified":
            # 修改后恢复：重置计数
            self._fail_counts[session_id] = 0
            logger.info(f"[中断] 🔄 会话 {session_id} 已修改后恢复，工作流继续执行")
        else:
            logger.warning(f"[中断] 未知审核动作: {action}")
            return False

        return True

    def is_interrupted(self, session_id: str) -> bool:
        """检查指定会话是否处于中断状态"""
        record = self._interrupts.get(session_id)
        if record is None:
            return False
        return record.status == "pending"

    def reset_fail_count(self, session_id: str) -> None:
        """重置指定会话的失败计数"""
        self._fail_counts[session_id] = 0

    def clear_interrupt(self, session_id: str) -> None:
        """清除指定会话的中断记录"""
        self._interrupts.pop(session_id, None)
        self._fail_counts.pop(session_id, None)

    def get_fail_count(self, session_id: str) -> int:
        """获取指定会话的连续失败次数"""
        return self._fail_counts.get(session_id, 0)


# ==================== 全局单例 ====================

_interrupt_handler_instance: Optional[InterruptHandler] = None


def get_interrupt_handler() -> InterruptHandler:
    """获取中断处理器单例"""
    global _interrupt_handler_instance
    if _interrupt_handler_instance is None:
        _interrupt_handler_instance = InterruptHandler()
    return _interrupt_handler_instance


if __name__ == "__main__":
    """本地测试"""
    logger.info("===== 中断处理器测试 =====")

    handler = get_interrupt_handler()

    # 模拟两次失败触发中断
    for i in range(3):
        triggered = handler.check_and_trigger(
            session_id="test_eval_session",
            eval_stage="retrieval",
            eval_result={"context_precision": 0.3, "context_recall": 0.4},
            fail_reasons=["ContextPrecision 低于阈值"],
        )
        logger.info(f"第 {i+1} 次检查: triggered={triggered}")

    # 查看中断
    interrupt = handler.get_interrupt("test_eval_session")
    if interrupt:
        logger.info(f"中断详情: id={interrupt['interrupt_id']}, status={interrupt['status']}")

    # 审核通过
    handler.review_interrupt("test_eval_session", "approved", "人工确认上下文正确，批准通过")
    logger.info(f"审核后状态: {handler.get_interrupt('test_eval_session')['status']}")

    # 检查是否还中断
    logger.info(f"是否仍中断: {handler.is_interrupted('test_eval_session')}")

    handler.clear_interrupt("test_eval_session")
    logger.info("===== 中断处理器测试完成 =====")
