"""
Context Compact 压缩引擎模块
功能：对超过指定轮数的对话历史进行智能压缩摘要，生成紧凑的对话摘要
核心流程：
  1. 轮数检测：检查历史对话是否超过阈值（首次 >=5轮 / 增量 >=3轮）
  2. 增量切片：增量模式下仅取上次压缩之后的新对话，避免 LLM 重复处理
  3. 提取历史：从 MongoDB 获取指定会话的全部历史记录
  4. 构造数据：按 user/assistant 交替格式组织历史文本
  5. LLM 压缩：调用大模型对增量对话进行语义摘要压缩
  6. 摘要拼接：增量模式下将新压缩内容追加到旧摘要后，形成完整压缩历史
  7. 结果缓存：将摘要结果缓存到 Redis（含 query/answer/compressed_history）
  8. 分析记录：生成压缩过程解析文件到 context_compact/ 目录

压缩策略：
  - 全量压缩（首次）：保留核心语义 + 保持对话脉络 + 实体完整性
  - 增量压缩（后续）：仅压缩新增对话，与旧摘要拼接，大幅减少 LLM 调用成本
  - 比例控制：压缩后长度控制原内容的 30%~50% 之间
"""
import json
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.conf.redis_config import redis_config
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import get_recent_messages, get_history_mongo_tool
from app.utils.path_util import PROJECT_ROOT


class ContextCompactEngine:
    """
    上下文压缩引擎
    职责：
      1. 检测并触发历史对话压缩
      2. 调用 LLM 生成压缩摘要
      3. 输出压缩结果和下阶段缓存所需数据

    使用方式：
      engine = ContextCompactEngine()
      result = engine.compact(session_id="xxx", history=history_list)
      # result = {"summary_query": "...", "summary_answer": "...", "compressed_history": "...", ...}
    """

    def __init__(self):
        """初始化压缩引擎（无状态，可复用）"""
        self.compact_dir = PROJECT_ROOT / "context_compact" / "analysis"
        # 确保分析文件目录存在
        try:
            self.compact_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"创建压缩分析目录失败: {e}")
            self.compact_dir = None

    def should_compact(
        self,
        history: List[Dict[str, Any]],
        last_compact_turn_count: int = 0
    ) -> bool:
        """
        判断是否需要执行压缩（支持增量压缩策略）
        策略：
          - 从未压缩过：会话轮数 >= compact_turn_threshold（默认 5）时首次触发
          - 已压缩过：  新增轮数 >= compact_incremental_threshold（默认 3）时再次触发
        参数：
            history: 历史对话列表（每条含 role/text 字段）
            last_compact_turn_count: 上次压缩时的会话轮数，0 表示从未压缩过
        返回：
            True=需要压缩, False=不需要
        """
        if not history:
            return False

        # 计算完整对话轮数（一对 user+assistant 算一轮）
        user_count = sum(1 for msg in history if msg.get("role") == "user")

        if last_compact_turn_count == 0:
            # 从未压缩过：使用首次触发阈值
            threshold = redis_config.compact_turn_threshold
            need = user_count >= threshold
            if need:
                logger.info(
                    f"[压缩检测] 首次触发：会话轮数({user_count}) >= 阈值({threshold})，触发压缩"
                )
            else:
                logger.debug(
                    f"[压缩检测] 会话轮数({user_count}) < 阈值({threshold})，跳过压缩"
                )
        else:
            # 已压缩过：使用增量阈值
            new_turns = user_count - last_compact_turn_count
            inc_threshold = redis_config.compact_incremental_threshold
            need = new_turns >= inc_threshold
            if need:
                logger.info(
                    f"[压缩检测] 增量触发：新增轮数({new_turns}) >= 增量阈值({inc_threshold})，"
                    f"当前总轮数({user_count})，上次压缩时({last_compact_turn_count})"
                )
            else:
                logger.debug(
                    f"[压缩检测] 新增轮数({new_turns}) < 增量阈值({inc_threshold})，"
                    f"当前总轮数({user_count})，跳过压缩"
                )

        return need

    def _format_history_for_llm(self, history: List[Dict[str, Any]]) -> Tuple[str, int]:
        """
        格式化历史对话为 LLM 可处理的文本
        输出格式：
          [用户]: 您好，请问烫金机怎么操作？
          [助手]: 您好！烫金机的操作步骤如下...
          ---
          [用户]: 温度怎么设置？
          [助手]: 建议设置 110℃...
        参数：
            history: 历史对话列表
        返回：
            (格式化文本, 总轮数)
        """
        lines = []
        turn_count = 0
        for msg in history:
            role = msg.get("role", "")
            text = msg.get("text", "").strip()
            if role == "user":
                turn_count += 1
                lines.append(f"[用户]: {text}")
            elif role == "assistant":
                lines.append(f"[助手]: {text}")
        return "\n".join(lines), turn_count

    def _slice_incremental_history(
        self,
        history: List[Dict[str, Any]],
        last_compact_turn_count: int
    ) -> List[Dict[str, Any]]:
        """
        从完整历史中截取增量部分：仅保留 last_compact_turn_count 之后的对话
        原理：遍历历史，统计 user 消息数，跳过前 last_compact_turn_count 轮，
              从第 last_compact_turn_count+1 轮开始保留
        参数：
            history: 完整历史对话列表
            last_compact_turn_count: 上次压缩时已覆盖的 user 轮数
        返回：
            增量历史列表（仅包含上次压缩之后的新对话）
        """
        if last_compact_turn_count <= 0:
            return history

        user_seen = 0
        for i, msg in enumerate(history):
            if msg.get("role") == "user":
                user_seen += 1
            if user_seen > last_compact_turn_count:
                # 找到了增量起点：从这里开始返回
                sliced = history[i:]
                new_user_count = sum(1 for m in sliced if m.get("role") == "user")
                logger.info(
                    f"[压缩引擎] 增量切片：全量 user={user_seen}轮，"
                    f"跳过前{last_compact_turn_count}轮，保留增量{new_user_count}轮"
                )
                return sliced

        # 兜底：没找到增量起点（理论上不会发生），返回空
        logger.warning(f"[压缩引擎] 增量切片未找到起点，last_compact={last_compact_turn_count}")
        return []

    def _extract_item_names(self, history: List[Dict[str, Any]]) -> List[str]:
        """
        从历史记录中提取涉及的商品名/实体名
        参数：
            history: 历史对话列表
        返回：
            商品名列表（去重）
        """
        item_names = set()
        for msg in history:
            names = msg.get("item_names", []) or []
            for name in names:
                if name and name.strip():
                    item_names.add(name.strip())
        return list(item_names)

    def compact(
        self,
        session_id: str,
        history: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
        last_compact_turn_count: int = 0,
        previous_compressed_history: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        执行对话历史压缩（核心方法）
        策略：
          - 首次压缩（last_compact_turn_count==0）：全量压缩全部历史
          - 增量压缩（last_compact_turn_count>0）：仅压缩上次压缩之后的新对话，
            然后与 previous_compressed_history 拼接，避免 LLM 重复处理旧内容
        参数：
            session_id: 会话 ID
            history: 历史对话列表（可选，不传则自动从 MongoDB 获取最近 50 条）
            force: 是否强制压缩（跳过轮数检测）
            last_compact_turn_count: 上次压缩时的会话轮数，0 表示首次
            previous_compressed_history: 上次压缩产生的完整 compressed_history 文本
        返回：
            压缩结果字典，失败或无需压缩返回 None
        """
        try:
            # 1. 获取历史记录（如果未传入则从 MongoDB 拉取）
            if history is None:
                history = get_recent_messages(session_id, limit=50)

            if not history:
                logger.warning(f"[压缩引擎] 会话 {session_id} 无历史记录，跳过压缩")
                return None

            # 2. 轮数检测（除非强制跳过）
            if not force and not self.should_compact(history, last_compact_turn_count):
                logger.debug(f"[压缩引擎] 会话 {session_id} 未达到压缩阈值，跳过")
                return None

            # 3. 增量切片：仅取上次压缩之后的新对话，避免 LLM 重复处理旧内容
            is_incremental = last_compact_turn_count > 0 and previous_compressed_history
            if is_incremental:
                target_history = self._slice_incremental_history(history, last_compact_turn_count)
                if not target_history:
                    logger.warning(f"[压缩引擎] 增量切片为空，跳过压缩")
                    return None
                logger.info(
                    f"[压缩引擎] 增量压缩模式：仅压缩新增部分，"
                    f"上次已压缩{last_compact_turn_count}轮，"
                    f"本次新增{sum(1 for m in target_history if m.get('role') == 'user')}轮"
                )
            else:
                target_history = history
                logger.info(f"[压缩引擎] 全量压缩模式：首次压缩全部历史")

            # 4. 格式化历史文本
            history_text, new_turn_count = self._format_history_for_llm(target_history)
            # 全量 user 轮数（从完整 history 统计，用于记录）
            total_turn_count = sum(1 for msg in history if msg.get("role") == "user")
            original_length = len(history_text)
            logger.info(f"[压缩引擎] 开始压缩会话 {session_id}，新增 {new_turn_count} 轮，长度 {original_length} 字符")

            # 5. 提取商品名（从全量历史提取，保证实体完整性）
            item_names = self._extract_item_names(history)

            # 6. 调用 LLM 压缩（仅压缩增量部分）
            logger.info(f"[压缩引擎] 调用 LLM 进行语义压缩...")
            compressed_new = self._call_llm_compact(history_text, new_turn_count)

            if not compressed_new:
                logger.warning(f"[压缩引擎] LLM 压缩返回空，使用原文截断作为降级")
                compressed_new = history_text[:1000] + "..." if len(history_text) > 1000 else history_text

            # 7. 拼接：增量模式下将新压缩内容追加到旧摘要后
            if is_incremental and previous_compressed_history:
                compressed_text = (
                    previous_compressed_history
                    + "\n\n---\n[后续对话]\n"
                    + compressed_new
                )
                logger.info(
                    f"[压缩引擎] 增量拼接完成：旧摘要{len(previous_compressed_history)}字 + "
                    f"新压缩{len(compressed_new)}字 = {len(compressed_text)}字"
                )
            else:
                compressed_text = compressed_new

            compressed_length = len(compressed_text)
            ratio = round((1 - compressed_length / max(original_length, 1)) * 100, 2)
            logger.info(f"[压缩引擎] 压缩完成: {original_length} → {compressed_length} 字符，压缩比 {ratio}%")

            # 8. 构建压缩结果（turn_count 记录全量轮数，供增量压缩标记位使用）
            summary_id = str(uuid.uuid4())[:8]
            result = {
                "summary_id": summary_id,
                "session_id": session_id,
                "summary_query": self._extract_core_question(history_text, compressed_text),
                "summary_answer": self._extract_core_answer(compressed_text),
                "compressed_history": compressed_text,
                "item_names": item_names,
                "turn_count": total_turn_count,
                "original_length": original_length,
                "compressed_length": compressed_length,
                "compression_ratio": ratio,
                "timestamp": datetime.now().timestamp(),
            }

            # 9. 生成压缩过程分析文件
            self._write_analysis_file(session_id, summary_id, result, history_text, is_incremental)

            logger.info(f"[压缩引擎] 会话 {session_id} 压缩成功，摘要ID={summary_id}")
            return result

        except Exception as e:
            logger.error(f"[压缩引擎] 压缩会话 {session_id} 失败: {str(e)}", exc_info=True)
            return None

    def _call_llm_compact(self, history_text: str, turn_count: int) -> str:
        """
        调用 LLM 对对话历史进行语义压缩
        参数：
            history_text: 格式化后的历史对话文本
            turn_count: 对话轮数
        返回：
            压缩后的文本
        """
        try:
            llm = get_llm_client()
            prompt = load_prompt(
                "context_compact",
                history_text=history_text,
                turn_count=str(turn_count)
            )
            response = llm.invoke(prompt)
            compressed = response.content.strip()
            return compressed
        except Exception as e:
            logger.error(f"[压缩引擎] LLM 压缩调用失败: {str(e)}")
            raise

    def _extract_core_question(self, history_text: str, compressed_text: str) -> str:
        """
        从压缩文本中提取核心问题摘要（取压缩文本的前 150 字作为问题代表）
        参数：
            history_text: 原始历史文本
            compressed_text: 压缩后的文本
        返回：
            核心问题摘要
        """
        # 从压缩文本中提取第一句话作为核心问题摘要
        lines = compressed_text.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            if len(first_line) > 10:
                return first_line[:150]
        # 降级：使用原始文本的前 100 字
        return history_text[:100] + "..." if len(history_text) > 100 else history_text

    def _extract_core_answer(self, compressed_text: str) -> str:
        """
        从压缩文本中提取核心答案摘要
        参数：
            compressed_text: 压缩后的文本
        返回：
            核心答案摘要（完整压缩文本本身即包含 Q&A 的精炼版本）
        """
        return compressed_text

    def _write_analysis_file(
        self,
        session_id: str,
        summary_id: str,
        result: Dict[str, Any],
        original_text: str,
        is_incremental: bool = False
    ) -> None:
        """
        生成压缩过程解析文件到 context_compact/analysis/ 目录
        文件内容：记录压缩的完整过程，包括原始长度、压缩策略、中间步骤等
        参数：
            session_id: 会话 ID
            summary_id: 摘要唯一标识
            result: 压缩结果字典
            original_text: 原始未压缩的对话文本
        """
        if self.compact_dir is None:
            logger.warning("[压缩引擎] 分析目录未创建，跳过写分析文件")
            return

        try:
            # 构建分析文件内容
            analysis = {
                "分析文件信息": {
                    "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "会话ID": session_id,
                    "摘要ID": summary_id,
                    "压缩引擎版本": "1.0.0",
                },
                "一、压缩触发条件": {
                    "触发机制": "首次 >= 5 轮触发全量压缩；之后每新增 >= 3 轮触发增量压缩（仅压缩新增部分并拼接旧摘要）",
                    "压缩模式": "增量压缩" if is_incremental else "全量压缩（首次）",
                    "当前会话轮数": f"{result['turn_count']} 轮",
                    "压缩阈值": f"{redis_config.compact_turn_threshold} 轮",
                    "是否达到阈值": result['turn_count'] >= redis_config.compact_turn_threshold,
                },
                "二、压缩前数据": {
                    "原始文本长度": f"{result['original_length']} 字符",
                    "原始文本预览": original_text[:500] + "..." if len(original_text) > 500 else original_text,
                    "涉及商品/实体": result['item_names'],
                },
                "三、压缩过程": {
                    "步骤1": "从 MongoDB 获取会话历史记录 (get_recent_messages)",
                    "步骤2": "格式化历史对话为 [用户]/[助手] 交替文本",
                    "步骤3": "调用 LLM (通义千问 qwen-flash) 进行语义压缩",
                    "步骤4": f"使用提示词模板: prompts/context_compact.prompt",
                    "步骤5": "压缩策略: 保留核心Q&A + 去除冗余寒暄 + 保持实体完整性",
                    "步骤6": "提取核心问题摘要和答案摘要",
                },
                "四、压缩结果": {
                    "压缩后文本长度": f"{result['compressed_length']} 字符",
                    "压缩比": f"{result['compression_ratio']}%",
                    "核心问题摘要": result['summary_query'],
                    "核心答案摘要预览": result['summary_answer'][:200] + "..." if len(result['summary_answer']) > 200 else result['summary_answer'],
                },
                "五、缓存策略": {
                    "缓存目标": "Redis Hash (qa:summary:{session_id}:{summary_id})",
                    "过期时间": f"{redis_config.cache_ttl} 秒 (24小时)",
                    "下次检索": "用户提问时先查 Redis 缓存，通过三级递进式相关性判断决定是否复用缓存答案",
                },
                "六、压缩评估": {
                    "是否保留核心信息": result['compression_ratio'] < 70,  # 压缩比小于 70% 认为保留了核心
                    "是否去除冗余": result['compressed_length'] < result['original_length'],
                    "下次检索建议": "建议使用 BGE-Reranker Cross-Encoder 相关性评分判断缓存命中",
                }
            }

            # 写入分析文件
            file_name = f"compact_{session_id}_{summary_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = self.compact_dir / file_name

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)

            logger.info(f"[压缩引擎] 分析文件已生成: {file_path}")

        except Exception as e:
            logger.warning(f"[压缩引擎] 写分析文件失败: {str(e)}")


# ==================== 全局单例管理 ====================

_compact_engine_instance: Optional[ContextCompactEngine] = None


def get_compact_engine() -> ContextCompactEngine:
    """获取压缩引擎单例"""
    global _compact_engine_instance
    if _compact_engine_instance is None:
        _compact_engine_instance = ContextCompactEngine()
    return _compact_engine_instance


if __name__ == "__main__":
    """本地测试：验证压缩引擎功能"""
    logger.info("===== 压缩引擎本地测试 =====")

    engine = get_compact_engine()

    # 构造测试历史数据（6轮对话，超过5轮阈值）
    mock_history = []
    for i in range(1, 7):
        mock_history.append({"role": "user", "text": f"这是第{i}轮用户提问，关于HAK 180烫金机的操作。"})
        mock_history.append({"role": "assistant", "text": f"这是第{i}轮助手的回答，详细说明了烫金机的操作步骤{i}。"})

    # 执行压缩
    result = engine.compact(
        session_id="test_compact_session",
        history=mock_history,
        force=True
    )

    if result:
        logger.info(f"压缩测试通过: 摘要ID={result['summary_id']}, "
                    f"原始={result['original_length']}→压缩={result['compressed_length']}, "
                    f"压缩比={result['compression_ratio']}%")
        logger.info(f"核心问题摘要: {result['summary_query']}")
        logger.info(f"检查分析文件是否生成: {engine.compact_dir}")
    else:
        logger.warning("压缩测试未返回结果")

    logger.info("===== 压缩引擎测试完成 =====")
