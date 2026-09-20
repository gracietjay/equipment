from app.utils.task_utils import *
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
import sys

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对） 分比例
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对） 分值
RERANK_GAP_ABS: float = 0.5

# Rerank节点（工作流入口）
def step_1_merge_docs(state):
    """
    阶段一：文档合并与标准化
    
    目标：将多路召回（本地知识库 + 联网搜索）的异构数据，统一合并为 Reranker 模型可处理的标准格式。
    
    输入来源：
    1. rrf_chunks (List[Dict]): 本地知识库检索结果（经 RRF 融合排序）。
       - 结构：包含 Milvus entity 信息的复杂字典或对象。
       - 关键字段：chunk_id, content, title/item_name。
    2. web_search_docs (List[Dict]): 联网搜索结果（经 MCP 搜索返回）。
       - 结构：包含搜索摘要的扁平字典。
       - 关键字段：snippet, title, url。
       
    输出结果 (List[Dict]):
    - 标准化文档列表，每项包含：
      - text: 用于重排序的核心文本（content 或 snippet）
      - title: 标题（用于增强语义或展示）
      - doc_id/chunk_id: 唯一标识（本地文档有，联网文档为 None）
      - url: 来源链接（本地为空，联网文档有）
      - source: 来源标记 ("local" 或 "web")
    """
    
    # 1. 提取输入源
    rrf_docs = state.get("rrf_chunks") or []
    web_docs = state.get("web_search_docs") or []
    
    logger.info(f"Step 1: 开始合并文档 - 本地RRF源: {len(rrf_docs)}条, 联网Web源: {len(web_docs)}条")
    doc_items = []
    # ---------------------------------------------------------
    # 2. 处理本地知识库文档 (rrf_chunks)
    # ---------------------------------------------------------
    for i, doc in enumerate(rrf_docs):
        # 简化：直接使用 dict(doc) 转换，如果 doc 本身是 dict 则无损，如果是对象则尝试转换
        # 由于上游 RRF 节点已经做了 _as_entity_list 处理，这里 doc 极大概率已经是纯字典
        # 因此可以移除繁琐的 try-except 和 entity 嵌套判断，直接取值
        
        # 兼容性处理：优先取 'entity' 字段（防守式编程），若无则视为 doc 本身即 entity
        # 注意：这里的 doc 应当已经是字典（由上游 _as_entity_list 保证）
        entity = doc.get("entity") if isinstance(doc, dict) and "entity" in doc else doc
        
        # 提取核心文本 (content)，这是重排序的依据
        # 如果不是字典或无 content，则跳过
        if not isinstance(entity, dict):
            logger.warning(f"本地文档格式异常 (index={i}): {type(entity)}")
            continue
            
        content = entity.get("content")
        if not content:
            # 仅在 debug 模式记录，避免生产环境日志刷屏
            logger.debug(f"跳过无内容文档 (index={i}, keys={list(entity.keys())})")
            continue

        # 提取元数据 (使用 .get 链式回退，简洁明了)
        doc_id = entity.get("chunk_id") or entity.get("id")
        title = entity.get("title") or entity.get("item_name") or ""

        # 组装标准化对象
        doc_items.append({
            "text": content,
            "doc_id": doc_id,
            "chunk_id": doc_id,  # 兼容旧逻辑保留字段
            "title": title,
            "url": "",
            "source": "local",
        })

    # ---------------------------------------------------------
    # 3. 处理联网搜索文档 (web_search_docs)
    # ---------------------------------------------------------
    for i, doc in enumerate(web_docs):
        # 兼容不同字段名：优先取 snippet (摘要)，其次 content
        text = (doc.get("snippet") or doc.get("content") or "").strip()
        url = (doc.get("url") or "").strip()
        title = (doc.get("title") or "").strip()
        
        if not text:
            logger.debug(f"跳过无内容联网结果 (index={i})")
            continue
            
        doc_items.append({
            "text": text,
            "doc_id": None, # 联网结果无固定 ID
            "chunk_id": None,
            "title": title,
            "url": url,
            "source": "web",
        })

    logger.info(f"Step 1: 文档合并完成，共输出 {len(doc_items)} 条标准化文档")
    return doc_items


def step_2_rerank_docs(state, doc_items):
    """
    阶段二：对文档进行重排序
    - 输入 doc_items：[{ text,doc_id}, ...]（由第一阶段产出）
    - 输出：在 state 中写入 reranked_docs（结构化列表）
    """
    question = state.get("rewritten_query") or state.get("original_query") or ""

    # 如果没有文档或问题，直接返回
    if not doc_items or not question:
        logger.warning("Step 2: 跳过重排序 (无文档或无问题)")
        return []

    logger.info(f"Step 2: 开始重排序 (Rerank), 待排序文档数: {len(doc_items)}")
    
    # 初始化重排序模型（这里以使用 BGE 重排序模型为例）
    texts = [x["text"] for x in doc_items]
    try:
        reranker = get_reranker_model()

        # 构建查询-文档对（必须是 str）
        """
           格式：列表，每个元素是二元元组 / 列表，严格遵循 (query, passage) 顺序（即你的「问题、答案」）：
             第 1 个元素（query）：用户的问题 / 检索词（如 “什么是 RRF 算法？”）；
             第 2 个元素（passage）：候选答案 / 待匹配文档（如你之前 RRF 融合后的文档内容）；
             支持单组匹配和批量匹配：
             # 单组匹配：1个问题+1个候选答案
             sentence_pairs = [("什么是RRF算法？", "RRF是倒数排名融合算法，用于多来源排序结果融合")]
             # 批量匹配：1个问题+多个候选答案（重排序核心场景，推荐）
             sentence_pairs = [
                   ("什么是RRF算法？", "RRF是倒数排名融合算法，用于多来源排序结果融合"),
                   ("什么是RRF算法？", "FP16是半精度推理，能降低模型显存占用"),
                   ("什么是RRF算法？", "FlagReranker是BGE重排序模型的封装类")
             ]
              注意：顺序不可颠倒（必须是「问题在前，答案在后」），模型对输入顺序有严格要求，颠倒会导致打分结果失真。
           2. 输出结果：scores 分数含义与格式
           格式：列表，元素为浮点数，列表长度与 sentence_pairs 完全一致，一一对应（第 n 个分数对应第 n 个 (问题，答案) 元组的相关性）；
           分数含义：数值越高，代表「问题」与「答案」的语义匹配度 / 相关性越强（BGE 重排序模型的分数无固定取值范围，核心看相对大小，用于排序即可）；
           核心用途：将分数与候选答案绑定，按分数降序排列，即可得到「与问题最相关→最不相关」的答案排序，实现重排序。
        """
        # 格式：列表，每个元素是二元元组 / 列表，严格遵循 (query, passage) 顺序
        sentence_pairs = [[question, t] for t in texts]
        # 计算相关性得分
        logger.info("Step 2: 正在计算相关性得分...")
        scores = reranker.compute_score(sentence_pairs)
        # 将得分与文档配对并排序（按 score 降序）
        scored_docs = []
        for item, text, score in zip(doc_items, texts, scores):
            # 保留两位小数便于日志查看
            score_val = float(score)
            scored_docs.append(
                {
                    "text": text,
                    "score": score_val,
                    "source": item.get("source") or "",
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                }
            )
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs
    except Exception as e:
        logger.error(f"Step 2: 重排序过程发生异常: {e}", exc_info=True)
        # 出错时降级：返回原始文档顺序，分数置为 0 或 None
        # 避免整个流程中断
        fallback_docs = [
            {
                "text": x.get("text"),
                "score": 0.0, # 降级分数
                "source": x.get("source") or "",
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "url": x.get("url") or "",
                "title": x.get("title") or "",
            }
            for x in doc_items
        ]
        # 在这里我们不直接修改 state，而是返回结果让主流程处理
        # 但为了兼容原有逻辑（虽然函数签名是返回 scored_docs），我们记录异常并抛出或返回降级结果
        # 这里选择返回降级结果，保证流程继续
        return fallback_docs

def step_3_topk(scored_docs):
    """
    阶段三：动态 TopK（最多 10）
    基于 scored_docs（已按 score 降序排序）进行智能截断，
    核心逻辑：结合固定上下限+断崖阈值判断，避免机械取前N条，保留语义相关的连续文档集合
    :param scored_docs: 列表，元素为带score的文档字典，已按score降序排列，格式如[{"doc": 文档对象, "score": 相关性分数}, ...]
    :return: 列表，动态截断后的TopK文档列表，数量≤10
    """
    # 硬上限：最多取前10条，取全局常量与实际文档数的较小值（避免索引越界）
    # 注：max_topk从全局常量读取，不依赖外部状态，保证逻辑一致性
    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK  # 硬下限：至少保留的文档数量（全局常量配置）
    gap_ratio = RERANK_GAP_RATIO  # 相对断崖阈值：分数下降的相对比例阈值（全局常量配置）
    gap_abs = RERANK_GAP_ABS      # 绝对断崖阈值：分数下降的绝对差值阈值（全局常量配置）

    # 1) 断崖截断核心逻辑：从min_topk之后开始检测分数断崖，出现则提前截断
    topk = max_topk  # 默认值：无断崖时取满硬上限（最多10条）
    # 仅当实际可取值超过硬下限时，才触发断崖检测（否则直接取满min_topk）
    if topk > min_topk:
        # 遍历范围：从min_topk-1到max_topk-2（索引从0开始），检测相邻两个文档的分数差
        # 例：min_topk=3，max_topk=10 → 遍历i=2,3,4,5,6,7,8（对应第3~9条文档，检测与下一条的差距）
        for i in range(min_topk - 1, max_topk - 1):
            s1 = scored_docs[i].get("score")  # 当前位置文档的分数
            s2 = scored_docs[i + 1].get("score")  # 下一个位置文档的分数

            gap = s1 - s2  # 计算相邻文档的分数绝对差距（因已降序，gap≥0）
            # 计算相对差距：绝对差距 / 当前文档分数（+1e-6避免除数为0/极小值，防止程序报错）
            # 1e-6 是 Python 中科学计数法的写法，等价于 0.000001（10 的负 6 次方，也就是百万分之一）。
            rel = gap / (abs(s1) + 1e-6)
            # 触发断崖截断条件：绝对差距≥绝对阈值 OR 相对差距≥相对阈值
            # 满足任一条件，说明下一条文档相关性骤降，截断在当前位置
            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(f"Step 3: 触发断崖截断 @ index={i} (Score {s1:.4f} -> {s2:.4f}, Gap={gap:.4f})")
                topk = i + 1  # 最终取前i+1条（索引转实际数量，如i=2 → 取前3条）
                break  # 触发截断后立即退出循环，不再检测后续位置

    # 按最终计算的topk值，截取前topk条文档
    topk_docs = scored_docs[:topk]
    
    logger.info(f"Step 3: 截断完成，保留前 {len(topk_docs)} 条文档 (TopK={topk})")
    
    if topk_docs:
        preview = ", ".join([f"{d.get('chunk_id') or 'Web'}({d.get('score'):.3f})" for d in topk_docs[:3]])
        logger.debug(f"Step 3: Top3 文档预览: {preview}")
        
    # 返回动态TopK处理后的文档列表
    return topk_docs


def node_rerank(state):
  """
  Rerank节点
  对检索到的文档进行重新排序，提高相关性
  """
  logger.info("---Rerank (重排序) 节点开始处理---")
  add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

  # 阶段一：合并文档
  doc_items = step_1_merge_docs(state)
  # 阶段二：对文档进行重排序
  scored_docs = step_2_rerank_docs(state, doc_items)
  # 阶段三：动态 TopK
  topk_docs = step_3_topk(scored_docs)
  
  logger.info(f"Rerank 节点处理结束, 最终输出 {len(topk_docs)} 条文档")

  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  return {"reranked_docs": topk_docs}


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> 启动 node_rerank 本地测试")
    print("="*50)
    
    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8},
        {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1} # 预期低分
    ]
    
    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"} # 预期低分
    ]
    
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？", # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])
        
        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)
        
        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")
            
        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后
        
        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("="*50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
