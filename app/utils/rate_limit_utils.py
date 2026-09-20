# app/utils/rate_limit_utils.py
import time
from typing import Deque
from app.core.logger import logger  # 复用项目全局logger


def apply_api_rate_limit(
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 60
) -> None:
    """
    通用滑动窗口API速率限制器（抽离为公共工具）
    核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
    :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
    :param max_requests: 速率限制窗口内的最大允许请求次数
    :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
    :return: None，超出限制时会阻塞等待
    """
    current_time = time.time()

    # 1. 清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
    while request_times and current_time - request_times[0] >= window_seconds:
        request_times.popleft()

    # 2. 窗口内请求数达上限，计算并阻塞等待剩余时间
    if len(request_times) >= max_requests:
        # 计算需要等待的时长（窗口总时长 - 最早请求已存在的时长）
        sleep_duration = window_seconds - (current_time - request_times[0])
        if sleep_duration > 0:
            logger.debug(f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")
            time.sleep(sleep_duration)
            # 等待后更新当前时间，重新清理过期请求（避免等待期间有请求过期）
            current_time = time.time()
            while request_times and current_time - request_times[0] >= window_seconds:
                request_times.popleft()

    # 3. 记录当前请求时间戳，加入滑动窗口队列
    request_times.append(current_time)
    logger.debug(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")