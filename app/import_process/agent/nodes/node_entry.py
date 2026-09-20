import os
import sys
from os.path import splitext

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task

def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph知识库导入工作流 - 入口节点
    核心职责：初始化参数校验 | 自动判断文件类型(PDF/MD) | 设置解析开关 | 提取业务标识
    入参：ImportGraphState - 必须包含 local_file_path(文件路径)、task_id(任务ID)
    出参：ImportGraphState - 新增/更新 is_pdf_read_enabled/is_md_read_enabled/pdf_path/md_path/file_title
    执行链路：__start__ → 本节点 → route_after_entry(条件路由) → 对应解析节点/流程终止
    """

    # 动态获取函数名避免硬编码
    func_name = sys._getframe().f_code.co_name

    # 节点启动日志，打印当前工作流状态
    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")

    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)


    # 1. 核心参数提取与非空校验
    document_path = state.get("local_file_path", "")
    if not document_path:
        logger.error(f"【{func_name}】核心参数缺失：工作流状态中未配置local_file_path，文件路径为空")
        return state

    # 2. 根据文件后缀判断类型，设置对应解析开关
    if document_path.endswith(".pdf"):
        logger.info(f"【{func_name}】文件类型校验通过：{document_path} → PDF格式，开启PDF解析流程")
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = document_path
    elif document_path.endswith(".md"):
        logger.info(f"【{func_name}】文件类型校验通过：{document_path} → MD格式，开启MD解析流程")
        state["is_md_read_enabled"] = True
        state["md_path"] = document_path
    else:
        logger.warning(f"【{func_name}】文件类型校验失败：{document_path} → 不支持的格式，仅支持.pdf/.md")

    # 3. 提取文件无后缀纯名称，作为全局业务标识
    file_name = os.path.basename(document_path)
    state["file_title"] = splitext(file_name)[0]
    logger.info(f"【{func_name}】文件业务标识提取完成：file_title = {state['file_title']}")

    # 结束：记录节点运行状态
    add_done_task(state["task_id"], func_name)

    # 节点完成日志，打印当前工作流状态
    logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态：{format_state(state)}")

    return state

if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    node_entry(test_state1)

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    node_entry(test_state2)

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")