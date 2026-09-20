# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
import os
import sys
from typing import List, Dict, Any, Tuple

# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500


def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict]]:
    """
    步骤 1: 接收并校验流程输入（商品名称识别的前置数据处理）
    核心作用：
        1. 从流程状态中提取文件标题、文本切片核心数据
        2. 做多层空值兜底，避免后续流程因空值报错
        3. 基础数据类型校验，保证下游流程输入有效性
    依赖的状态数据（上游节点产出）：
        - state["file_title"]: 上游提取的文件标题（优先使用）
        - state["file_name"]: 原始文件名（file_title为空时兜底）
        - state["chunks"]: 文本切片列表（每个切片为字典，含title/content等字段）
    返回值：
        Tuple[str, List[Dict]]: (处理后的文件标题, 校验后的文本切片列表)
    """
    # 多层兜底获取文件标题：优先file_title → 其次file_name → 空字符串
    file_title = state.get("file_title", "") or state.get("file_name", "")
    # 获取文本切片列表：空值时返回空列表，避免后续遍历报错
    chunks = state.get("chunks") or []

    # 二次兜底：file_title仍为空时，尝试从第一个有效切片中提取
    if not file_title:
        if chunks and isinstance(chunks[0], dict):
            file_title = chunks[0].get("file_title", "")
            logger.warning("state中无有效file_title，已从第一个切片中提取兜底标题")

    # 空值日志提示：文件标题为空时不中断流程，仅记录警告
    if not file_title:
        logger.warning("state中缺少file_title和file_name，后续大模型识别可能精度下降")

    # 数据类型校验：确保chunks为有效非空列表，否则返回空列表
    if not isinstance(chunks, list) or not chunks:
        logger.warning("state中chunks为空或非列表类型，无法进行商品名称识别")
        return file_title, []

    logger.info(f"步骤1：输入校验完成，获取到{len(chunks)}个有效文本切片")
    return file_title, chunks


def step_2_build_context(chunks: List[Dict], k: int = DEFAULT_ITEM_NAME_CHUNK_K, max_chars: int = CONTEXT_TOTAL_MAX_CHARS) -> str:
    """
    步骤 2: 构造大模型商品名称识别的标准化上下文
    核心作用：
        1. 限制切片数量：仅取前k个切片，避免上下文过长
        2. 限制字符长度：单切片+总上下文双重字符限制，适配大模型输入上限
        3. 格式化内容：带序号的结构化格式，提升大模型识别精度
        4. 过滤无效切片：跳过空内容/非字典类型切片，保证上下文有效性
    参数说明：
        chunks: 文本切片列表（每个元素为字典，需包含"title"和"content"键）
        k: 最大取片数，默认5个（可通过配置调整）
        max_chars: 上下文总字符数上限，默认2500（适配大模型输入限制）
    返回值：
        str: 格式化后的上下文字符串（直接传给大模型，空切片时返回空字符串）
    """
    # 空切片直接返回空字符串，无需后续处理
    if not chunks:
        return ""

    # 存储格式化后的切片片段，保证上下文结构化
    parts: List[str] = []
    # 统计已拼接字符数，用于控制总长度不超限
    total_chars = 0

    # 遍历前k个切片，避免上下文过长
    for idx, chunk in enumerate(chunks[:k]):
        # 跳过非字典类型切片，防止键取值报错
        if not isinstance(chunk, dict):
            logger.debug(f"第{idx+1}个切片非字典类型，已过滤")
            continue

        # 提取切片标题和内容，去首尾空格，过滤无效字符
        chunk_title = chunk.get("title", "").strip()
        chunk_content = chunk.get("content", "").strip()

        # 标题和内容均为空，跳过该无效切片
        if not (chunk_title or chunk_content):
            logger.debug(f"第{idx+1}个切片为空白内容，已过滤")
            continue

        # 单切片内容截断：防止单个切片内容过长占满上下文
        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]
            logger.debug(f"第{idx+1}个切片内容过长，已截断至{SINGLE_CHUNK_CONTENT_MAX_LEN}字符")

        # 结构化格式化切片：带序号+标题+内容，提升大模型识别效率
        piece = f"【切片{idx + 1}】\n标题：{chunk_title} \n内容：{chunk_content}"
        parts.append(piece)
        # 累计字符数，包含分隔符
        total_chars += len(piece)

        # 总字符数超限时立即停止拼接，避免大模型输入超限
        if total_chars > max_chars:
            logger.info(f"上下文总字符数即将超限（{max_chars}），已停止拼接后续切片")
            break

    # 用空行分隔切片片段，拼接为最终上下文，最后一次去重空格
    context = "\n\n".join(parts).strip()
    # 最终二次截断，确保绝对不超限
    final_context = context[:max_chars]
    logger.info(f"步骤2：上下文构建完成，最终长度{len(final_context)}字符")
    return final_context


def step_3_call_llm(file_title: str, context: str) -> str:
    """
    步骤 3: 调用大模型实现商品名称/型号精准识别
    核心逻辑：
        1. 上下文为空 → 直接返回file_title（兜底，无需调用大模型）
        2. 上下文非空 → 加载标准化prompt模板，构建大模型对话消息
        3. 调用大模型后对返回结果做清洗，过滤无效字符
        4. 大模型返回空/调用异常 → 均返回file_title兜底，保证流程不中断
    核心特性：
        - 提示词解耦：通过load_prompt加载本地模板，无需硬编码
        - 格式兼容：兼容不同LLM客户端返回格式，防止属性报错
        - 异常兜底：全异常捕获，大模型服务不可用时不影响主流程
    参数：
        file_title: 处理后的文件标题（异常/空值时的兜底值）
        context: 步骤2构建的结构化切片上下文（大模型识别的核心依据）
    返回值：
        str: 清洗后的商品名称（异常/空值时返回原始file_title）
    """
    logger.info("开始执行步骤3：调用大模型识别商品名称")

    # 上下文为空时，直接返回文件标题，跳过大模型调用
    if not context:
        logger.warning("上下文为空，跳过大模型调用，直接使用文件标题作为商品名称")
        return file_title

    try:
        # 加载商品名称识别prompt模板，动态传入文件标题和上下文
        human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
        # 加载系统提示词，定义大模型角色（商品识别专家，仅返回纯结果）
        system_prompt = load_prompt("product_recognition_system")
        logger.debug(f"大模型调用提示词构建完成，系统提示词长度{len(system_prompt)}，人类提示词长度{len(human_prompt)}")

        # 获取大模型客户端：json_mode=False，要求返回纯文本而非JSON格式
        llm = get_llm_client(json_mode=False)
        if not llm:
            logger.error("大模型客户端获取失败，使用文件标题兜底")
            return file_title

        # 标准化构建大模型对话消息：SystemMessage定义角色 + HumanMessage传递业务请求
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        # 调用大模型并获取返回结果
        resp = llm.invoke(messages)

        # 兼容不同LLM客户端返回格式：优先取content字段，无则返回空字符串
        item_name = getattr(resp, "content", "").strip()
        # 清洗返回结果：过滤空格、换行、回车、制表符等无效字符
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

        # 清洗后结果为空，使用文件标题兜底
        if not item_name:
            logger.warning("大模型返回空内容，使用文件标题作为商品名称兜底")
            return file_title

        logger.info(f"步骤3：大模型识别商品名称成功，结果为：{item_name}")
        return item_name

    # 捕获所有异常：大模型调用超时、网络错误、格式错误等，均不中断主流程
    except Exception as e:
        logger.error(f"步骤3：大模型调用失败，原因：{str(e)}", exc_info=True)
        # 异常时返回文件标题兜底，保证流程继续执行
        return file_title

def step_4_update_chunks(state: ImportGraphState, chunks: List[Dict], item_name: str):
    """
    步骤 4: 回填商品名称到流程状态和所有文本切片
    核心作用：
        1. 全局状态更新：将item_name存入state，供下游所有节点直接使用
        2. 切片数据补全：为每个切片添加item_name字段，保证数据一致性
        3. 状态同步：更新state中的chunks，确保切片修改全局生效
    设计思路：
        所有切片关联同一商品名称，保证后续向量入库、检索时的维度一致性
    参数：
        state: 流程状态对象（ImportGraphState），全局数据载体
        chunks: 校验后的文本切片列表（步骤1输出）
        item_name: 步骤3识别并清洗后的商品名称
    """
    # 将商品名称存入全局状态，供下游节点调用
    state["item_name"] = item_name
    # 遍历所有切片，为每个切片添加商品名字段，保证数据全链路一致
    for chunk in chunks:
        chunk["item_name"] = item_name
    # 同步更新state中的切片列表，确保修改全局生效
    state["chunks"] = chunks
    logger.info(f"步骤4：商品名称回填完成，共为{len(chunks)}个切片添加item_name字段，值为：{item_name}")

def step_5_generate_vectors(item_name: str) -> Tuple[Any, Any]:
    """
    步骤 5: 为商品名称生成BGE-M3稠密+稀疏双向量（Milvus向量检索核心）
    核心说明：
        - 稠密向量（dense_vector）：BGE-M3固定1024维，记录文本深层语义信息
        - 稀疏向量（sparse_vector）：变长键值对，记录文本关键词/特征位置信息
    依赖工具：
        generate_embeddings：封装BGE-M3模型，批量生成双向量，兼容单条/批量输入
    参数：
        item_name: 步骤3识别的商品名称（非空，空值时直接返回空向量）
    返回值：
        Tuple[Any, Any]: (稠密向量列表, 稀疏向量字典)，空值/异常时返回(None, None)
    """
    logger.info(f"开始执行步骤5：为商品名称[{item_name}]生成BGE-M3双向量")

    # 商品名称为空，直接返回空向量，跳过模型调用
    if not item_name:
        logger.warning("商品名称为空，跳过向量生成，返回空向量")
        return None, None

    try:
        # 调用向量生成工具：传入列表支持批量生成，单条数据仍用列表保证格式统一
        vector_result = generate_embeddings([item_name])

        # 向量生成结果非空，才进行后续解析
        if vector_result and "dense" in vector_result and "sparse" in vector_result:
            # 稠密向量解析：取批量结果第一个，为Python列表（Milvus存储要求）
            dense_vector = vector_result["dense"][0]
            # 稀疏向量解析：取批量结果第一个，CSR矩阵解析为字典格式
            sparse_vector = vector_result["sparse"][0]
            logger.info("步骤5：BGE-M3稠密+稀疏向量生成成功")
        else:
            logger.warning("步骤5：向量生成工具返回空结果，无法提取双向量")
            dense_vector, sparse_vector = None, None

    # 捕获所有异常：模型加载失败、向量生成超时、格式错误等
    except Exception as e:
        logger.error(f"步骤5：向量生成失败，原因：{str(e)}", exc_info=True)
        dense_vector, sparse_vector = None, None

    return dense_vector, sparse_vector


def step_6_save_to_milvus(state: ImportGraphState, file_title: str, item_name: str, dense_vector, sparse_vector):
    """
    步骤 6: 将商品名称、文件标题、双向量持久化到Milvus向量数据库
    核心逻辑：
        1. 配置校验：检查Milvus连接地址和集合名配置，缺失则跳过
        2. 客户端获取：获取单例Milvus客户端，连接失败则跳过
        3. 集合初始化：无集合则创建（定义Schema+索引），有集合则直接使用（保留原有配置）
        4. 幂等性处理：删除同名商品数据，避免重复存储
        5. 数据插入：构造符合Schema的数据，非空向量才添加
        6. 集合加载：插入后强制加载集合，确保数据立即可查/Attu可见
    参数：
        state: 流程状态对象，用于最终状态同步
        file_title: 处理后的文件标题
        item_name: 识别后的商品名称（主键去重依据）
        dense_vector: 步骤5生成的稠密向量（1024维列表）
        sparse_vector: 步骤5生成的稀疏向量（字典格式）
    """
    # 从环境变量读取Milvus核心配置，与MilvusConfig配置类保持一致
    milvus_uri = os.environ.get("MILVUS_URL")
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")

    # 配置缺失校验：任一配置为空则跳过Milvus存储，记录警告
    if not all([milvus_uri, collection_name]):
        logger.warning("Milvus配置缺失（MILVUS_URL/ITEM_NAME_COLLECTION），跳过数据保存")
        return

    logger.info(f"开始执行步骤6：将商品名称[{item_name}]保存到Milvus集合[{collection_name}]")

    try:
        # 获取Milvus单例客户端，连接失败则直接返回
        client = get_milvus_client()
        if not client:
            logger.error("无法获取Milvus客户端（连接失败），跳过数据保存")
            return

        # 集合初始化：不存在则创建（定义Schema+索引），存在则直接使用
        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Milvus集合[{collection_name}]不存在，开始创建Schema和索引")
            # 创建集合Schema：自增主键+动态字段，适配灵活的数据存储
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            # 添加自增主键字段：INT64类型，唯一标识每条数据
            schema.add_field(
                field_name="pk",
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=True
            )
            # 添加文件标题字段：VARCHAR类型，最大长度65535，适配长标题
            schema.add_field(
                field_name="file_title",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            # 添加商品名字段：VARCHAR类型，最大长度65535，去重依据
            schema.add_field(
                field_name="item_name",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            # 添加稠密向量字段：FLOAT_VECTOR，1024维（BGE-M3固定维度）
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=1024
            )
            # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR，变长
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR
            )

            # 构建索引参数：为向量字段创建索引，提升检索性能
            index_params = client.prepare_index_params()
            # 优化版稠密向量索引：HNSW + COSINE (恢复最佳性能配置)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                # HNSW (Hierarchical Navigable Small World) 是目前性能最好、最常用的基于图的索引，检索速度极快，精度极高。
                index_type="HNSW",
                # 使用 COSINE 作为稠密向量相似度计算方式
                metric_type="COSINE",
                # M: 图中每个节点的最大连接数(常用16-64)
                # efConstruction: 构建索引时的搜索范围(越大建索引越慢，但精度越高，常用100-200)
                params={"M": 16, "efConstruction": 200}
            )

            # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+IP，关闭量化保证精度
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                # 稀疏倒排索引 专门为稀疏向量（比如文本的 TF-IDF 向量、关键词权重向量，特点是大部分元素为 0，只有少数维度有值）设计的倒排索引，是稀疏向量检索的标配索引类型。
                index_type="SPARSE_INVERTED_INDEX",
                # IP（内积，Inner Product）如果向量是 “文本语义向量 + 关键词权重”，长度代表文本与主题的关联强度，此时用 IP 能同时体现 “语义匹配度” 和 “关联强度”。
                metric_type="IP",
                #DAAT_MAXSCORE 是稀疏检索的高效算法，quantization="none" 保证稀疏向量权重无损失；normalize=是否归一化。
                params = {"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
            )

            # 创建集合：Schema + 索引参数
            client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
            logger.info(f"Milvus集合[{collection_name}]创建成功，包含Schema和向量索引")

        # 幂等性处理：删除同名商品数据，避免重复存储（核心：先加载集合才能删除）
        clean_item_name = (item_name or "").strip()
        if clean_item_name:
            client.load_collection(collection_name=collection_name)
            # 商品名称转义，防止特殊字符导致过滤表达式解析失败
            safe_item_name = escape_milvus_string(clean_item_name)
            filter_expr = f'item_name=="{safe_item_name}"'
            # 执行删除操作
            client.delete(collection_name=collection_name, filter=filter_expr)
            logger.info(f"Milvus幂等性处理完成，已删除集合中[{clean_item_name}]的历史数据")

        # 构造插入Milvus的数据：基础字段+非空向量字段
        data = {
            "file_title": file_title,
            "item_name": item_name
        }
        # 稠密向量非空才添加，避免空值入库报错
        if dense_vector is not None:
            data["dense_vector"] = dense_vector
        # 稀疏向量非空则归一化后添加，保证检索准确性
        if sparse_vector is not None:
            data["sparse_vector"] = normalize_sparse_vector(sparse_vector)

        # 插入数据：列表格式支持批量插入，单条数据保持格式统一
        client.insert(collection_name=collection_name, data=[data])
        # 插入后强制加载集合，确保数据立即可查、Attu可视化界面可见
        client.load_collection(collection_name=collection_name)

        # 最终同步商品名称到全局状态
        state["item_name"] = item_name
        logger.info(f"步骤6：商品名称[{item_name}]成功存入Milvus集合[{collection_name}]，数据：{list(data.keys())}")

    # 捕获所有Milvus操作异常：连接中断、入库失败、索引错误等，不中断主流程
    except Exception as e:
        logger.error(f"步骤6：数据存入Milvus失败，原因：{str(e)}", exc_info=True)

def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    【核心节点】商品主体名称识别（node_item_name_recognition）
    整体流程：提取输入→构建上下文→大模型识别→回填数据→生成向量→存入Milvus
    核心目的：利用大模型从文档切片中精准识别商品/主体名称，并生成双路向量（稠密+稀疏）存入数据库
    后续扩展点：支持多主体识别、增加商品属性提取、对接其他向量库等
    :param state: 项目状态字典（ImportGraphState），必须包含chunks/file_title/task_id
    :return: 更新后的状态字典，新增item_name键，且chunks列表中每个元素新增item_name字段
    """
    # 初始化当前节点信息，用于任务监控和日志溯源
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【商品名称识别】{node_name}")
    # 将当前节点加入运行中任务，更新全局任务状态
    add_running_task(state.get("task_id", ""), node_name)

    try:
        # ===================================== 步骤1：提取并校验输入数据 =====================================
        # 作用：从状态字典提取文件标题和切片列表，校验数据完整性
        # 输出：文件标题、切片列表；若无切片则抛出异常或终止
        file_title, chunks = step_1_get_inputs(state)
        if not chunks:
            logger.warning(f">>> 节点执行警告：{node_name}（无有效切片数据），跳过识别")
            return state

        # ===================================== 步骤2：构建大模型识别上下文 =====================================
        # 作用：截取前N个切片的内容，拼接成大模型可阅读的上下文，用于辅助识别
        # 输出：拼接后的上下文字符串
        context = step_2_build_context(chunks)

        # ===================================== 步骤3：调用大模型识别商品名称 =====================================
        # 作用：构造Prompt，调用LLM从上下文和标题中提取最核心的商品名称
        # 输出：识别出的商品名称字符串（如 "iPhone 15 Pro"）
        item_name = step_3_call_llm(file_title, context)

        # ===================================== 步骤4：回填商品名称到状态和切片 =====================================
        # 作用：将识别结果写入状态字典，并同步更新到每一个Chunk对象的元数据中
        # 输出：状态字典新增item_name，chunks列表被就地修改
        step_4_update_chunks(state, chunks, item_name)

        # ===================================== 步骤5：生成双路向量（稠密+稀疏） =====================================
        # 作用：调用BGE-M3模型，为商品名称生成稠密语义向量和稀疏关键词向量
        # 输出：dense_vector（List[float]）、sparse_vector（Dict[int, float]）
        dense_vector, sparse_vector = step_5_generate_vectors(item_name)

        # ===================================== 步骤6：存入Milvus向量数据库 =====================================
        # 作用：将商品名称及其双路向量存入Milvus的 item_names 集合，用于后续检索
        # 输出：无返回值，数据已持久化
        step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        # 节点执行完成日志
        logger.info(f">>> 核心节点执行完成：【商品名称识别】{node_name}，识别结果：{item_name}，已存入Milvus")

    except Exception as e:
        # 全局异常捕获：保证节点执行失败不崩溃整个流程，记录详细错误日志便于排查
        logger.error(f">>> 核心节点执行失败：【商品名称识别】{node_name}，错误信息：{str(e)}", exc_info=True)
        # 可选：失败时设置默认值或标记状态
        state["item_name"] = "未知商品"

    # 返回更新后的状态（供下游节点使用）
    return state

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()