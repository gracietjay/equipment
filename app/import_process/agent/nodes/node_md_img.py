import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


# 步骤1：初始化MD核心数据，获取内容、文件路径、图片文件夹路径
def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    从全局状态中提取并初始化MD处理所需核心数据
    :param state: 导入流程全局状态对象
    :return: 三元组(MD文件内容, MD文件路径对象, 图片文件夹路径对象)
    :raise FileNotFoundError: 当状态中无有效MD文件路径时抛出
    """
    md_file_path = state["md_path"]
    # 校验MD文件路径有效性
    if not md_file_path:
        raise FileNotFoundError(f"全局状态中无有效MD文件路径：{state['md_path']}")

    path_obj = Path(md_file_path)
    # 优先使用状态中已存在的MD内容，无则从文件读取
    if not state["md_content"]:
        with open(path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        logger.debug(f"从文件读取MD内容完成，文件大小：{len(md_content)} 字符")
    else:
        md_content = state["md_content"]
        logger.debug(f"从全局状态获取MD内容完成，内容大小：{len(md_content)} 字符")

    # 图片文件夹固定为MD文件同级的images目录
    images_dir = path_obj.parent / "images"
    return md_content, path_obj, images_dir


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def find_image_in_md(md_content: str, image_filename: str, context_len: int = 100) -> List[Tuple[str, str]]:
    """
    查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
    :param md_content: MD文件完整内容
    :param image_filename: 图片文件名（含后缀）
    :param context_len: 上下文截取长度，默认前后各100字符
    :return: 上下文列表，每个元素为(上文, 下文)元组，无匹配则返回空列表
    """
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")
    results = []

    for m in pattern.finditer(md_content):
        start, end = m.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        # 打印图片上下文，便于调试
        logger.debug(f"图片[{image_filename}]匹配到引用，上文：{pre_text.strip()}")
        logger.debug(f"图片[{image_filename}]匹配到引用，下文：{post_text.strip()}")
        results.append((pre_text, post_text))
    if not results:
        logger.debug(f"MD内容中未找到图片[{image_filename}]的引用")
    return results

# 步骤2：扫描图片文件夹，筛选MD中实际引用的支持格式图片
def step_2_scan_images(md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
    :param md_content: MD文件完整内容
    :param images_dir: 图片文件夹路径对象
    :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
    """
    targets = []
    # 遍历图片文件夹所有文件
    # 遍历图片文件夹所有文件
    for image_file in os.listdir(images_dir):
        # 过滤非支持格式的图片
        if not is_supported_image(image_file):
            logger.debug(f"图片格式不支持，跳过：{image_file}")
            continue
        # 组装图片完整路径
        img_path = str(images_dir / image_file)
        # 查找图片在MD中的引用上下文
        context_list = find_image_in_md(md_content, image_file)
        # 过滤MD中未引用的图片
        if not context_list:
            logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")
            continue
        # 组装待处理图片元数据，取第一个匹配的上下文
        targets.append((image_file, img_path, context_list[0]))
        logger.info(f"图片加入待处理列表：{image_file}")
    logger.info(f"图片扫描完成，共筛选出待处理图片：{len(targets)} 张")
    return targets


def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片文件编码为Base64字符串（用于多模态大模型输入）
    :param image_path: 图片本地完整路径
    :return: 图片的Base64编码字符串（UTF-8解码）
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"图片Base64编码完成，文件：{image_path}，编码后长度：{len(base64_str)}")
    return base64_str

def summarize_image(image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
    """
    调用多模态大模型生成图片内容摘要（适配LangChain工具类，复用项目统一LLM客户端）
    生成的摘要用于Markdown图片标题，严格控制50字以内中文描述
    :param image_path: 图片本地完整路径
    :param root_folder: 文档所属文件夹/主名，为大模型提供上下文
    :param image_content: 图片在MD中的上下文元组，格式(上文文本, 下文文本)
    :return: 图片内容摘要（异常时返回默认值"图片描述"）
    """
    # 将图片编码为Base64，适配多模态大模型输入要求
    base64_image = encode_image_to_base64(image_path)
    try:
        # 1. 获取项目统一LLM客户端（自动缓存，传入多模态模型名）
        lvm_client = get_llm_client(model=lm_config.lv_model)

        # 加载并渲染提示词（核心：传入所有占位符对应的变量）
        prompt_text = load_prompt(
            name="image_summary",  # 提示词文件名（不带.prompt）
            root_folder=root_folder,  # 对应{root_folder}
            image_content=image_content  # 对应{image_content[0]}、{image_content[1]}
        )

        # 2. 构造LangChain标准多模态HumanMessage（兼容千问/OpenAI等视觉模型）
        messages = [
            HumanMessage(
                content=[
                    # 文本提示词：携带上下文，限定摘要规则
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    # 多模态核心：Base64编码图片数据
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]

        # 3. LangChain标准调用：invoke方法（工具类已封装超时/重试等参数）
        response = lvm_client.invoke(messages)

        # 4. 解析响应（LangChain统一返回content字段，统一格式无需多层解析）
        summary = response.content.strip().replace("\n", "")
        logger.info(f"图片摘要生成成功：{image_path}，摘要：{summary}")
        return summary

    except LangChainException as e:
        logger.error(f"图片摘要生成失败（LangChain框架异常）：{image_path}，错误信息：{str(e)}")
        return "图片描述"
    except Exception as e:
        logger.error(f"图片摘要生成失败（系统异常）：{image_path}，错误信息：{str(e)}")
        return "图片描述"

def step_3_generate_summaries(doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                              requests_per_minute: int = 9) -> Dict[str, str]:
    """
    步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
    :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
    :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
    :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
    """
    summaries = {}
    request_times = deque()  # 外部初始化请求时间队列，跨循环复用

    for img_file, image_path, context in targets:
        # 直接调用抽离的公共工具方法，参数和原逻辑完全一致
        apply_api_rate_limit(request_times, requests_per_minute, window_seconds=60)
        logger.debug(f"开始生成图片摘要：{image_path}")
        summaries[img_file] = summarize_image(image_path, root_folder=doc_stem, image_content=context)

    logger.info(f"图片摘要批量生成完成，共处理{len(summaries)}张图片")
    return summaries


def clean_minio_directory(minio_client: Minio, prefix: str) -> None:
    """
    幂等性清理MinIO指定目录下的所有旧文件，防止重名文件内容混淆和垃圾文件堆积
    幂等性：多次调用结果一致，无文件时不报错
    :param minio_client: 初始化完成的MinIO客户端对象
    :param prefix: MinIO目录前缀（要清理的目录路径）
    """
    try:
        # 列出指定前缀下的所有对象（递归遍历子目录）
        objects_to_delete = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=prefix,
            recursive=True
        )
        # 构造删除对象列表
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]

        if delete_list:
            logger.info(f"开始清理MinIO旧文件，待删除文件数：{len(delete_list)}，目录：{prefix}")
            # 批量删除对象
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            # 遍历删除错误信息，记录异常
            for error in errors:
                logger.error(f"MinIO文件删除失败：{error}")
        else:
            logger.debug(f"MinIO目录无旧文件，无需清理：{prefix}")
    except Exception as e:
        logger.error(f"MinIO目录清理失败：{prefix}，错误信息：{str(e)}")


def upload_images_batch(minio_client: Minio, upload_dir: str, targets: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[
    str, str]:
    """
    批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
    :param minio_client: 初始化完成的MinIO客户端对象
    :param upload_dir: MinIO上传根目录
    :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
    """
    urls = {}
    for img_file, img_path, _ in targets:
        # 构造MinIO对象名称
        object_name =  f"{upload_dir}/{img_file}"
        logger.debug(f"构造MinIO对象名称完成：{object_name}")
        # 上传单张图片并获取URL
        """
        := 是 Python 3.8+ 引入的海象运算符（Walrus Operator），核心作用是 **「表达式内赋值 + 结果判断」一体化 **：
        在执行判断、循环等逻辑的同一个表达式中，完成变量赋值和赋值结果的使用 / 判断，替代传统「先赋值、后判断」的两行代码，让逻辑更简洁。
        """
        if img_url := upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url
    logger.info(f"图片批量上传完成，成功上传{len(urls)}/{len(targets)}张图片")
    return urls

def upload_to_minio(minio_client: Minio, local_path: str, object_name: str) -> str | None:
    """
    将单张本地图片上传至MinIO对象存储，并返回公网可访问URL
    :param minio_client: 初始化完成的MinIO客户端对象
    :param local_path: 图片本地完整路径
    :param object_name: MinIO中要存储的对象名称（带目录）
    :return: 图片MinIO访问URL（上传失败返回None）
    """
    try:
        logger.info(f"开始上传图片至MinIO：本地路径={local_path}，MinIO对象名={object_name}")
        # 上传本地文件至MinIO（fput_object：文件流上传，适合大文件）
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,  # MinIO存储桶名（从配置读取）
            object_name=object_name,  # MinIO对象名称
            file_path=local_path,  # 本地文件路径
            # 自动推断图片Content-Type（如image/png、image/jpeg）
            # 入参：文件路径字符串（可带目录，如/a/b/test.jpg、demo.tar.gz）；
            # 返回值：元组(root, ext)，其中：
            # root：文件主名（含目录，去掉最后一个后缀的完整部分）；
            # ext：文件后缀（以.开头，仅包含最后一个扩展名，如.jpg、.gz，无后缀则为空字符串""）；
            # 关键规则：仅识别 ** 最后一个.** 作为后缀分隔符，多后缀文件仅拆分最后一个（如test.tar.gz拆分为("test.tar", ".gz")）。
            content_type=f"image/{os.path.splitext(local_path)[1][1:]}"
        )

        # 处理路径特殊字符，避免URL解析错误
        # 假设原始 object_name 是：图片\logo.png
        # 替换后变成：图片%5Clogo.png
        # 这个字符串是URL 合法格式，所有服务器 / 浏览器都能正确识别；
        # MinIO 接收到 %5C 后，会自动解析回 \，保证对象名的正确性；
        # 后续通过 URL 访问时，%5C 会被正确解码，不会出现路径错误。
        object_name = object_name.replace("\\", "%5C")
        # 根据配置选择HTTP/HTTPS协议
        protocol = "https" if minio_config.minio_secure else "http"
        # 构造MinIO基础访问URL
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"
        # 拼接完整图片访问URL base_url 后面带 / 中间直接两个字符串拼接即可
        img_url = f"{base_url}{object_name}"
        logger.info(f"图片上传成功，访问URL：{img_url}")
        return img_url
    except Exception as e:
        logger.error(f"图片上传MinIO失败：{local_path}，错误信息：{str(e)}")
        return None

def merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """
    合并图片摘要字典和URL字典，过滤掉上传失败无URL的图片
    :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
    :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
    :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
    """
    image_info = {}
    # 遍历摘要字典，仅保留有对应URL的图片
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    logger.info(f"图片摘要与URL合并完成，有效图片信息{len(image_info)}条")
    return image_info


def process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """
    核心功能：替换MD内容中的本地图片引用为MinIO远程引用
    替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
    :param md_content: 原始MD文件内容
    :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
    :return: 替换后的新MD内容
    """
    for img_filename, (summary, new_url) in image_info.items():
        # 正则匹配MD图片标签，忽略大小写，兼容不同路径写法
        # 正则规则：![任意描述](任意路径+图片文件名+任意后缀)
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)",
            re.IGNORECASE
        )
        # 替换匹配内容：使用新摘要作为图片描述，新URL作为图片路径
        # - 如果你的 summary 和 new_url 是完全可控的纯文本（不含反斜杠） ：这两种写法确实 一模一样 。
        # - 如果你想写出“防御性代码”（Defensive Code），防止未来某天被特殊字符坑 ：请坚持使用 Lambda 写法 。它是最稳健、最安全的做法。
        # md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)
        md_content = pattern.sub( f"![{summary}]({new_url})", md_content)
        logger.debug(f"完成MD图片引用替换：{img_filename} → {new_url}")

    logger.info(f"MD文件图片引用替换完成，共替换{len(image_info)}处图片引用")
    logger.debug(f"替换后MD内容：{md_content[:500]}..." if len(md_content) > 500 else f"替换后MD内容：{md_content}")
    return md_content

def step_4_upload_and_replace(minio_client: Minio, doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                              summaries: Dict[str, str], md_content: str) -> str:
    """
    步骤4：核心流程-图片上传MinIO + 合并摘要&URL + 替换MD图片引用
    完整流程：清理MinIO旧目录 → 批量上传新图片 → 合并摘要和URL → 替换MD内容
    :param minio_client: 初始化完成的MinIO客户端对象
    :param doc_stem: 文档文件名（不含后缀），作为MinIO上传子目录名（按文档隔离）
    :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
    :param md_content: 原始MD文件内容
    :return: 图片引用替换后的新MD内容
    """
    # 构造MinIO上传目录：配置根目录 + 文档主名（去除空格，避免路径问题）
    minio_img_dir = minio_config.minio_img_dir
    upload_dir = f"{minio_img_dir}/{doc_stem}".replace(" ", "")

    # 步骤1：清理该文档对应的MinIO旧目录，保证幂等性
    clean_minio_directory(minio_client, upload_dir)
    # 步骤2：批量上传图片至MinIO，获取URL映射
    urls = upload_images_batch(minio_client, upload_dir, targets)
    # 步骤3：合并图片摘要和URL，过滤上传失败的图片
    # Dict[str, Tuple[str, str]]  键：图片文件名，值：(摘要, URL)元组
    image_info = merge_summary_and_url(summaries, urls)
    # 步骤4：替换MD内容中的本地图片引用为MinIO远程引用
    if image_info:
        md_content = process_md_file(md_content, image_info)

    return md_content


def step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
    """
    步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
    新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
    :param origin_md_path: 原始MD文件完整路径
    :param md_content: 处理后的新MD内容
    :return: 新MD文件的完整路径
    """
    # 构造新文件路径：替换原后缀为 _new.md
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

    # 写入新MD内容（覆盖写入，若文件已存在则更新）
    with open(new_md_file_name, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")
    return new_md_file_name


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    MD文件图片处理核心节点 - 五步法完成图片全流程处理
    核心流程：
    1. 初始化获取MD内容、文件路径、图片文件夹路径
    2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片
    3. 调用多模态大模型为图片生成内容摘要
    4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
    5. 备份原MD文件，保存处理后的新MD文件并更新状态
    :param state: 导入流程全局状态对象，包含task_id、md_path、md_content等核心参数
    :return: 更新后的全局状态对象（md_content/md_path为处理后新值）
    """
    # 记录当前运行任务，用于任务监控和状态追踪
    add_running_task(state["task_id"], sys._getframe().f_code.co_name)

    # 步骤1：初始化数据，获取MD核心信息
    md_content, path_obj, images_dir = step_1_get_content(state)
    state["md_content"] = md_content

    # 无图片文件夹，直接跳过所有图片处理逻辑
    if not images_dir.exists():
        logger.info(f"图片文件夹不存在，跳过图片处理：{images_dir.absolute()}")
        return state

    # 初始化MinIO客户端，失败则终止流程
    minio_client = get_minio_client()
    if not minio_client:
        logger.warning("MinIO客户端初始化失败，已跳过图片处理全流程")
        return state
    
    # 步骤2：扫描并筛选MD中引用的支持格式图片
    # (image_file, img_path, context_list[0])
    targets = step_2_scan_images(md_content, images_dir)
    if not targets:
        logger.info("未检测到MD中引用的支持格式图片，跳过后续处理")
        return state

    # 步骤3：调用多模态大模型生成图片摘要（修复原代码传参错误：使用文件主名而非MD内容）
    summaries = step_3_generate_summaries(path_obj.stem, targets)

    # 步骤4：上传图片至MinIO，替换MD图片路径并填充摘要
    new_md_content = step_4_upload_and_replace(minio_client, path_obj.stem, targets, summaries, md_content)
    state["md_content"] = new_md_content

    # 步骤5：备份并保存新MD文件，更新状态中的文件路径
    new_md_file_name = step_5_backup_new_md_file(state['md_path'], new_md_content)
    state["md_path"] = new_md_file_name
    logger.info(f"MD图片处理完成，新文件已保存：{new_md_file_name}")

    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")