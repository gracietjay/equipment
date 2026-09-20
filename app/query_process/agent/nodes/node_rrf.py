import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


# RRF节点
def _as_entity_list(state_list) -> List[Dict[str, Any]]:
    """
    将上游节点输出统一规整为 entity dict 列表。
    兼容：
    - dict: {"entity": {..属性名和对应的字.}, "distance": ...} 或直接就是 {...}
    - pymilvus Hit: 不是 dict，但通常支持 hit.get("entity") 或 hit.entity
    - 其他：当作 chunk_id
    """
    out: List[Dict[str, Any]] = []
    for doc in (state_list or []):
        if not doc:
            continue
        
        final_ent = {}
        
        # 情况A: doc 是 Pymilvus 的 Hit 对象 (具有 entity 属性)
        # Hit 对象结构通常是: id=xxx, distance=xxx, entity={field1: val1, ...}
        # 这里的 id 是 Milvus 内部的主键 ID (int64 或 str)
        if hasattr(doc, "entity") and hasattr(doc, "id"):
            # 1. 提取 entity 中的业务字段 (如 content, item_name, chunk_id 等)
            # 注意: doc.entity 可能是一个 Entity 对象，也可能直接是 dict
            entity_content = doc.entity
            if hasattr(entity_content, "to_dict"):
                 final_ent = entity_content.to_dict()
            elif isinstance(entity_content, dict):
                 final_ent = entity_content.copy()
            else:
                 # 尝试直接作为 dict 访问 (某些版本 sdk)
                 try:
                     final_ent = dict(entity_content)
                 except:
                     pass
            
            # 2. 补充最外层的 id 和 distance
            # 优先保留 entity 内部已有的 chunk_id/id，如果没有，则把外层的 id 补进去
            if "id" not in final_ent and "chunk_id" not in final_ent:
                final_ent["id"] = doc.id
            
            # 补充 distance (score)
            if hasattr(doc, "distance"):
                final_ent["score"] = doc.distance

        # 情况B: doc 已经是字典 (模拟数据或已处理数据)
        elif isinstance(doc, dict):
             # 尝试获取 entity 字段 (嵌套结构 {"entity": {...}, "id": ...})
             if "entity" in doc:
                 ent = doc["entity"]
                 if isinstance(ent, dict):
                     final_ent = ent.copy()
                 # 尝试从外层补充 id/score
                 if "id" in doc and "id" not in final_ent:
                     final_ent["id"] = doc["id"]
                 if "distance" in doc:
                     final_ent["score"] = doc["distance"]
             else:
                 # 扁平结构，直接使用
                 final_ent = doc

        # 情况C: 其他对象 (尝试 get 方法)
        elif hasattr(doc, "get"):
             ent = doc.get("entity") or doc
             if isinstance(ent, dict):
                 final_ent = ent
        
        # 最终校验：必须是非空字典
        if final_ent and isinstance(final_ent, dict):
            out.append(final_ent)
            
    return out


def reciprocal_rank_fusion(
        source_weights: list,
        k: int = 60,
        max_results: int = None,
) -> List[tuple]:
    """
    通用带权重的RRF算法实现
    :param source_weights:  列表，每个元素是(来源文档列表, 权重)的元组
                            例如: [([doc1, doc2], 1.0), ([doc2, doc3], 0.8)]
    :param k:     RRF 常数，默认 60。用于平滑排名影响，避免高排名文档占据过大优势。
    :param max_results: 只返回前 N 个，None 表示全部
    :return:      [(元素, RRF 得分), ...] 按得分降序排列
    """
    # score_map: 记录 chunk_id 到 RRF 累加得分的映射
    score_map = {}
    # chunk_map: 记录 chunk_id 到文档实体对象的映射，用于最终返回
    chunk_map = {}

    # 1. 遍历所有来源，计算每个文档的 RRF 分数
    # source_weights 结构: [(doc_list, weight), ...]
    for docs, weight in source_weights:
        # enumerate(docs, start=1): 获取排名 (rank)，从 1 开始
        for rank, item in enumerate(docs, start=1):
            # 获取文档唯一标识 ID
            # Milvus 设计上把主键字段在 API 层面统一叫 id，不管你在 schema 里定义的字段名是 pk、id 还是其他
            # 这是为了保持 API 兼容性：无论用户怎么命名主键，SDK 都用 id 来指代 “这条数据的唯一标识”
            # 你在向量数据库 UI 里看到的 pk 是表结构定义名，而代码里拿到的 id 是API 返回的统一主键别名
            chunk_id = item.get("chunk_id") or item.get("id")
            
            if not chunk_id:
                # 如果找不到 ID，记录警告并跳过，避免程序崩溃
                logger.warning(
                    f"RRF Warning: item missing chunk_id/id: {list(item.keys()) if isinstance(item, dict) else item}")
                continue

            # RRF 核心公式: score += weight * (1 / (k + rank))
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))
            
            # 只记录第一次遇到的文档实体对象
            chunk_map.setdefault(chunk_id, item)

    # 2. 将结果转换为列表并排序
    merged = []
    for chunk_id, score in score_map.items():
        doc_item = chunk_map[chunk_id]
        # [(chunk, score), (chunk, score)...]
        merged.append((doc_item, score))
    
    # 按分数降序排序 (得分越高越靠前)
    merged.sort(key=lambda x: x[1], reverse=True)
    
    # 3. 截断结果
    if max_results is not None:
        merged = merged[:max_results]
        
    return merged


def node_rrf(state):
    """
    RRF (Reciprocal Rank Fusion) 倒数排名融合节点
    
    功能：
    将来自不同检索源（如 Embedding 检索、HyDE 检索、知识图谱检索等）的结果进行融合排序。
    RRF 是一种无需训练的算法，仅根据文档在不同列表中的排名来计算最终得分。
    
    步骤：
    1. 提取各路检索结果：从 state 中获取 embedding_chunks 和 hyde_embedding_chunks。
    2. 结果标准化：将不同格式的检索结果统一转换为包含 chunk_id 的实体列表。
    3. 设置权重：为不同来源分配权重（当前配置：Embedding=1.0, HyDE=1.0）。
    4. 执行 RRF：计算融合分数并重新排序。
    5. 结果截断：保留 Top K 个结果。
    6. 更新状态：将融合后的结果存入 state["rrf_chunks"]。
    """
    logger.info("---RRF (倒数排名融合) 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 第一步：获取上游检索节点返回的文档
    # 上游检索节点（Milvus hybrid_search）返回的通常是 hit 列表：
    #  {"entity": {...fields...}, "distance": ...}
    # RRF 需要使用 chunk_id 做去重与计分，因此这里必须保留 entity（而不是仅抽取 content 字符串）。
    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))

    logger.info(f"RRF 输入统计: Embedding源={len(embedding_chunks)}条, HyDE源={len(hyde_embedding_chunks)}条")
    
    # Debug 日志：打印部分 ID 以便核对
    if embedding_chunks:
        logger.debug(f"Embedding源 chunk_ids (前5个): {[c.get('chunk_id') for c in embedding_chunks[:5]]}")
    if hyde_embedding_chunks:
        logger.debug(f"HyDE源 chunk_ids (前5个): {[c.get('chunk_id') for c in hyde_embedding_chunks[:5]]}")

    # 第二步：为不同来源设置权重
    # 当前策略：两路召回权重相等，均为 1.0
    source_weights = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]

    # 第三步：应用带权重的RRF计算最终得分
    # k=60 是 RRF 算法的经典常数，max_results=10 限制最终召回数量
    rrf_res = reciprocal_rank_fusion(source_weights, k=60, max_results=10)

    # 第四步：解包结果，提取文档和分数
    rrf_chunks = [doc for doc, score in rrf_res]
    # 记录任务结束
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    return {"rrf_chunks": rrf_chunks}


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> 启动 node_rrf 本地测试")
    print("="*50)

    # 1. 构造假数据 (模拟真实数据库字段)
    # 模拟 Embedding 检索结果 
    mock_embedding_chunks = [
        {
            "id": "doc_1", 
            "pk": "pk_1", 
            "file_title": "操作手册_v1.pdf", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容1：打开电源开关...", 
            "score": 0.9
        },
        {
            "id": "doc_2", 
            "pk": "pk_2", 
            "file_title": "维修指南.pdf", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容2：遇到故障请联系...", 
            "score": 0.8
        },
        {
            "id": "doc_3", 
            "pk": "pk_3", 
            "file_title": "参数表.xlsx", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容3：电压220V...", 
            "score": 0.7
        }
    ]
    
    # 模拟 HyDE 检索结果 (包含 3 个文档，顺序不同，且有新文档 doc_4)
    mock_hyde_chunks = [
        {
            "id": "doc_3", 
            "pk": "pk_3", 
            "file_title": "参数表.xlsx", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容3：电压220V...", 
            "score": 0.85
        }, 
        {
            "id": "doc_1", 
            "pk": "pk_1", 
            "file_title": "操作手册_v1.pdf", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容1：打开电源开关...", 
            "score": 0.82
        }, 
        {
            "id": "doc_4", 
            "pk": "pk_4", 
            "file_title": "安全须知.docx", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容4：操作时请佩戴手套...", 
            "score": 0.75
        }
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks
    }

    try:
        # 运行节点
        result = node_rrf(mock_state)
        
        # 验证结果
        rrf_chunks = result.get("rrf_chunks", [])
        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={len(mock_embedding_chunks)}, HyDE={len(mock_hyde_chunks)}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)
        
        # 打印详细排名
        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            # 注意：返回结果中可能没有 chunk_id 字段，而是 id
            doc_id = doc.get('chunk_id') or doc.get('id')
            print(f"Rank {i}: ID={doc_id}, Title={doc.get('file_title')}, Content={doc.get('content')[:20]}...")

        # 验证预期逻辑：
        ids = [d.get("id") or d.get("chunk_id") for d in rrf_chunks]
        
        if "doc_1" in ids and "doc_3" in ids:
            print("\n[PASS] 交叉文档 (doc_1, doc_3) 成功融合保留")
        else:
            print("\n[FAIL] 交叉文档丢失")
            
        if len(ids) == 4:
            print("[PASS] 并集数量正确 (3+3-2重叠=4)")
        else:
            print(f"[FAIL] 并集数量错误: 期望4, 实际{len(ids)}")
            
        print("="*50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
