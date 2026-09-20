import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())


def step_3_extract_info(query: str, history: List[Dict]) -> Dict:
    """
    利用LLM从当前问题以及历史会话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）
    若商品名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
    :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
    :param history: 列表[字典] - 近期会话历史
    :return: 字典 - 提取结果，格式：{"item_names": [], "rewritten_query": ""}
    """
    logger.info("Step 3: 开始提取信息 (LLM)")
    
    # 1. 初始化准备
    client = get_llm_client(json_mode=True)
    
    # 构造历史对话文本
    history_text = ""
    for msg in history:
        history_text += f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"
    
    logger.info(f"Step 3: 历史上下文构建完成，长度: {len(history_text)} 字符")

    # 2. 加载提示词
    try:
        # 使用关键字参数传递，避免参数位置错误
        prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
        logger.debug(f"Step 3: 提示词加载成功，Prompt长度: {len(prompt)}")
    except Exception as e:
        logger.error(f"Step 3: 加载提示词失败: {e}")
        return {"item_names": [], "rewritten_query": query}

    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt)
    ]

    try:
        logger.info("Step 3: 正在调用 LLM 进行提取...")
        response = client.invoke(messages)
        content = response.content
        logger.debug(f"Step 3: LLM 原始响应: {content}")

        # 清理 Markdown 代码块
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        
        result = json.loads(content)
        
        # 健壮性检查
        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = query
            
        logger.info(f"Step 3: 提取结果解析成功 - 商品名: {result['item_names']}, 重写问题: {result['rewritten_query']}")
        return result

    except Exception as e:
        logger.error(f"Step 3: LLM 提取或解析失败: {e}")
        return {"item_names": [], "rewritten_query": query}


def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict]:
    """
    对提取的 item_names 进行向量化并在 Milvus 中进行混合搜索
    """
    logger.info(f"Step 4: 开始向量化检索，目标商品: {item_names}")
    results = []
    
    client = get_milvus_client()
    if not client:
        logger.error("Step 4: 无法连接到 Milvus")
        return results

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logger.error("Step 4: 环境变量中未找到 ITEM_NAME_COLLECTION")
        return results

    try:
        logger.info("Step 4: 正在生成 Embedding (Dense + Sparse)...")
        embeddings = generate_embeddings(item_names)
        logger.info(f"Step 4: 向量生成完成，开始 Milvus 搜索 (Collection: {collection_name})")

        for i, name in enumerate(item_names):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                # 构造混合搜索请求
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )

                # 执行混合搜索
                # 权重调整为 0.8 (Dense) / 0.2 (Sparse) 以优化评分
                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2), 
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"]
                )

                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        entity = hit.get("entity") or {}
                        item_name = entity.get("item_name")
                        score = hit.get("distance")
                        
                        if item_name:
                            matches.append({
                                "item_name": item_name,
                                "score": score
                            })
                            logger.debug(f"Step 4: '{name}' 匹配项: {item_name} (Score: {score:.4f})")

                results.append({
                    "extracted_name": name,
                    "matches": matches
                })
                logger.info(f"Step 4: 商品 '{name}' 检索完成，找到 {len(matches)} 个匹配项")

            except Exception as inner_e:
                logger.error(f"Step 4: 处理商品 '{name}' 时出错: {inner_e}")
                results.append({"extracted_name": name, "matches": []})

    except Exception as e:
        logger.error(f"Step 4: 向量化或搜索过程发生全局错误: {e}")

    return results


def step_5_align_item_names(query_results: List[Dict]) -> Dict:
    """
    根据 Milvus 搜索评分，对齐商品名，生成「确认商品名」和「候选商品名」
    """
    logger.info("Step 5: 开始对齐商品名 (Score Analysis)")
    
    confirmed_item_names = []
    options = []

    for res in query_results:
        extracted_name = res.get("extracted_name", "").strip()
        matches = res.get("matches", []) or []
        
        if not matches:
            logger.info(f"Step 5: '{extracted_name}' 无匹配结果")
            continue

        # 按分数降序
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 打印详细评分日志辅助调试
        top_matches_log = ", ".join([f"{m['item_name']}({m['score']:.3f})" for m in matches[:3]])
        logger.info(f"Step 5: '{extracted_name}' Top匹配: {top_matches_log}")

        # 筛选
        high = [m for m in matches if m.get("score", 0) > 0.85]
        mid = [m for m in matches if m.get("score", 0) >= 0.6]

        # 规则 A: 单个高置信度
        if len(high) == 1:
            confirmed_name = high[0].get("item_name")
            confirmed_item_names.append(confirmed_name)
            logger.info(f"Step 5: 规则A命中 (Single High) -> 确认: {confirmed_name}")
            continue

        # 规则 B: 多个高置信度
        if len(high) > 1:
            picked = None
            # 优先匹配同名
            if extracted_name:
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        logger.info(f"Step 5: 规则B命中 (Exact Match in High) -> 确认: {picked.get('item_name')}")
                        break
            
            # 否则取最高分
            if not picked:
                picked = high[0]
                logger.info(f"Step 5: 规则B命中 (Highest Score) -> 确认: {picked.get('item_name')}")

            confirmed_item_names.append(picked.get("item_name"))
            continue

        # 规则 C: 无高置信度，取中置信度候选
        if len(mid) > 0:
            current_options = [m.get("item_name") for m in mid[:5]]
            options.extend(current_options)
            logger.info(f"Step 5: 规则C命中 (Mid Confidence) -> 添加候选: {current_options}")
            continue
        
        logger.info(f"Step 5: 规则D命中 (Low Confidence) -> 无匹配")

    result = {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options))
    }
    logger.info(f"Step 5: 对齐结果: {result}")
    return result


def step_6_check_confirmation(state: Dict, align_result: Dict, session_id: str, history: List[Dict], rewritten_query: str) -> Dict:
    """
    检查对齐结果，更新 State
    """
    logger.info("Step 6: 检查确认状态并更新 State")
    
    # 健壮性处理
    if align_result is None:
        align_result = {}

    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])

    # 分支 A: 有确认商品名
    if confirmed:
        logger.info(f"Step 6: [分支A] 存在确认商品名: {confirmed}")
        
        # 更新历史消息中的 item_names
        ids_to_update = []
        for msg in history:
            if not msg.get("item_names"):
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))
        
        if ids_to_update:
            logger.info(f"Step 6: 更新 {len(ids_to_update)} 条历史消息的关联商品名")
            update_message_item_names(ids_to_update, confirmed)

        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            del state["answer"]
        return state

    # 分支 B: 有候选商品名
    if options:
        logger.info(f"Step 6: [分支B] 存在候选商品名: {options}")
        options_str = "、".join(options[:3])
        answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
        state["answer"] = answer
        state["item_names"] = []
        return state

    # 分支 C: 无结果
    logger.info("Step 6: [分支C] 无确认也无候选")
    state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
    state["item_names"] = []
    return state


def step_7_write_history(state: Dict, session_id: str, history: List[Dict], rewritten_query: str, message_id: str) -> Dict:
    """
    写入最终历史记录
    """
    logger.info("Step 7: 写入会话历史")
    
    # 如果有助手回答（分支 B/C），写入助手消息
    if state.get("answer"):
        logger.info("Step 7: 保存助手回答")
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=state["answer"],
            rewritten_query="",
            item_names=[]
        )

    # 更新用户消息（关联 rewrite_query 和 item_names）
    logger.info(f"Step 7: 更新用户消息 (ID: {message_id})")
    save_chat_message(
        session_id=session_id,
        role="user",
        text=state["original_query"],
        rewritten_query=rewritten_query,
        item_names=state.get("item_names", []),
        message_id=message_id
    )

    return state


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    主节点函数：商品名称确认流程
    """
    logger.info(">>> node_item_name_confirm: 开始处理")
    
    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    # 标记任务开始
    add_running_task(session_id, "node_item_name_confirm", is_stream)

    # 1. 获取历史记录
    history = get_recent_messages(session_id, limit=10)
    logger.info(f"Node: 获取到 {len(history)} 条历史消息")

    # 2. 保存用户当前消息 (初始保存，后续 step 7 会更新)
    message_id = save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))
    logger.debug(f"Node: 用户消息已初始保存, ID: {message_id}")

    # 3. 提取信息
    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", original_query)

    # 合并近期历史中的 item_names（滑动窗口：最近 N 轮用户发言）
    # node_cache_check 已做过窗口提取，这里作为兜底：若 state 未带则从本地 history 提取
    RECENT_ROUND_WINDOW = 3
    history_item_names = state.get("history_item_names") or []
    if not history_item_names and history:
        # 兜底：从本地历史中取最近 N 轮 user 消息的 item_names
        user_messages = [msg for msg in history if msg.get("role") == "user"]
        recent_user_rounds = user_messages[-RECENT_ROUND_WINDOW:]
        history_item_names = list(set(
            n for msg in recent_user_rounds
            for n in (msg.get("item_names") or []) if n and str(n).strip()
        ))
        if history_item_names:
            logger.info(f"本地 history 滑动窗口提取 item_names "
                        f"(窗口={RECENT_ROUND_WINDOW}轮): {history_item_names}")

    if history_item_names:
        # 合并去重：LLM 从当前 query 提取的 + 近期滑动窗口内的
        merged = list(set(item_names + history_item_names))
        logger.info(f"合并近期历史 item_names: LLM={item_names}, 滑动窗口={history_item_names}, 合并={merged}")
        item_names = merged
    
    # 更新 State 中的 rewrite_query
    state["rewritten_query"] = rewritten_query

    align_result = {}

    # 4. & 5. 如果有提取到商品名，进行搜索和对齐（Milvus向量查询实体，模型提取的实体不一定和我们Milvus的完全相同）
    if len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: 未提取到商品名，跳过向量检索")

    # 6. 检查确认状态
    state = step_6_check_confirmation(state, align_result, session_id, history, rewritten_query)

    # 7. 写入最终历史
    final_state = step_7_write_history(state, session_id, history, rewritten_query, message_id)

    # 将 history 存入 state，供后续节点（如 node_answer_output）使用
    final_state["history"] = history

    # 标记任务完成
    add_done_task(session_id, "node_item_name_confirm", is_stream)
    
    logger.info(f"Node: 处理结束, Final State Item Names: {final_state.get('item_names')}")
    return final_state


if __name__ == "__main__":
    # 测试代码块
    print("\n" + "="*50)
    print(">>> 启动 node_item_name_confirm 本地测试")
    print("="*50)
    
    # 模拟输入状态
    mock_state = {
        "session_id": "test_debug_session_001",
        "original_query": "HAK 180 烫金机多少钱？",  # 针对用户提到的具体 case
        "is_stream": False,
        "item_names": []
    }

    try:
        # 运行节点
        result = node_item_name_confirm(mock_state)
        
        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        print(f"Rewritten Query: {result.get('rewritten_query')}")
        print(f"Item Names: {result.get('item_names')}")
        print(f"Answer: {result.get('answer')}")
        print("="*50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
