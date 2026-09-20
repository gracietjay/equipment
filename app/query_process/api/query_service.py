from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app

# V3.0 新增：中断处理器
from app.eval.interrupt_handler import get_interrupt_handler


# 定义fastapi对象
app = FastAPI(title="query service",description="智库查询服务！（V3.0 集成Ragas评估+中断机制）")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 返回chat.html页面
@app.get("/chat.html")
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{chat_html_path}！")
    return FileResponse(chat_html_path)


# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


# 定义中断审核请求数据结构（V3.0 新增）
class InterruptReviewRequest(BaseModel):
    """中断审核请求数据结构"""
    session_id: str = Field(..., description="会话ID")
    action: str = Field(..., description="审核动作：approved(批准) / rejected(拒绝) / modified(修改后恢复)")
    comment: str = Field("", description="审核意见")


# 证明服务器启动即可
@app.get("/health")
async def health():
    return {"ok": True}


# 定义查询接口
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")

    default_state = {"original_query": user_query, "session_id": session_id, "is_stream": is_stream}
    try:
        # 执行查询流程图（V3.0 含检索评估+生成评估+中断机制）
        query_app.invoke(default_state)
        # 整体任务更新完成
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    is_stream = request.is_stream

    if is_stream:
        create_sse_queue(session_id)

    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")

    if is_stream:
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")

        # V3.0 检查是否触发了中断
        handler = get_interrupt_handler()
        interrupt = handler.get_interrupt(session_id)

        response_data = {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "done_list": [],
        }

        if interrupt and interrupt.get("status") == "pending":
            # 有中断待审核，通知前端
            response_data["interrupt"] = {
                "interrupt_id": interrupt["interrupt_id"],
                "eval_stage": interrupt["eval_stage"],
                "fail_reasons": interrupt["fail_reasons"],
                "fail_count": interrupt["fail_count"],
                "eval_result": interrupt["eval_result"],
            }
            response_data["message"] = "检索评估未达标，请审核后重试（POST /eval/review）"

        return response_data


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    print("调用流式/stream...")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


# ==================== V3.0 新增：中断审核接口 ====================

@app.get("/eval/interrupts", summary="获取所有待审核的中断")
async def get_pending_interrupts():
    """获取所有待审核的评估中断列表"""
    handler = get_interrupt_handler()
    pending = handler.get_pending_interrupts()
    return {
        "code": 200,
        "total": len(pending),
        "interrupts": pending,
    }


@app.get("/eval/interrupt/{session_id}", summary="获取指定会话的中断详情")
async def get_interrupt_detail(session_id: str):
    """获取指定会话的中断详情"""
    handler = get_interrupt_handler()
    interrupt = handler.get_interrupt(session_id)
    if interrupt is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 无中断记录")
    return {"code": 200, "interrupt": interrupt}


@app.post("/eval/review", summary="人工审核中断")
async def review_interrupt(request: InterruptReviewRequest):
    """
    人工审核评估中断
    审核后工作流可继续执行或终止

    参数：
        session_id: 会话ID
        action: approved(批准通过) / rejected(拒绝) / modified(修改后恢复)
        comment: 审核意见
    """
    session_id = request.session_id
    action = request.action
    comment = request.comment

    # 校验 action 值
    if action not in ("approved", "rejected", "modified"):
        raise HTTPException(status_code=400, detail="action 必须是 approved / rejected / modified 之一")

    handler = get_interrupt_handler()
    success = handler.review_interrupt(session_id, action, comment)

    if not success:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 无待审核的中断记录")

    action_map = {
        "approved": "已批准，工作流将继续执行",
        "rejected": "已拒绝，工作流已终止",
        "modified": "已修改后恢复，工作流将继续执行",
    }

    return {
        "code": 200,
        "message": f"审核完成：{action_map.get(action, action)}",
        "session_id": session_id,
        "action": action,
        "comment": comment,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
