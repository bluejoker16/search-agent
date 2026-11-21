"""
FastAPI 接口层：
  GET  /                 聊天页面
  GET  /api/health       健康检查
  POST /api/chat         SSE 流式问答（核心接口）
  POST /api/retrieve     只看检索片段，用来在前端展示 RAG 证据
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agent import build_agent, retriever

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="LangChain 1.0 RAG 智能问答助手")

# 进程级单例：模型、向量库、Agent 只初始化一次
agent = build_agent()


class ChatRequest(BaseModel):
    question: str
    thread_id: str = "demo-user"


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "agent_ready": agent is not None}


@app.post("/api/retrieve")
def retrieve(req: ChatRequest):
    """返回检索到的原始片段 —— 面试演示时打开它，能直观证明答案来自文档。"""
    docs = retriever.invoke(req.question)
    return {
        "docs": [
            {"source": d.metadata.get("source", "未知来源"), "content": d.page_content}
            for d in docs
        ]
    }


def extract_text(content) -> str:
    """兼容两种返回：普通模型给 str；推理模型(V4-Flash)给内容块列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


@app.post("/api/chat")
def chat(req: ChatRequest):
    if agent is None:
        return JSONResponse(
            {"error": "请先在 .env 中配置 LLM_API_KEY（硅基流动 Key），再重启服务"},
            status_code=503,
        )

    def event(payload: dict) -> str:
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    def generate():
        # thread_id 相同即为同一会话，checkpointer 会自动带上历史消息
        config = {"configurable": {"thread_id": req.thread_id}}
        sent = ""
        try:
            for state in agent.stream(
                {"messages": [{"role": "user", "content": req.question}]},
                config=config,
                stream_mode="values",
            ):
                messages = state.get("messages", [])
                if not messages:
                    continue
                last = messages[-1]
                if getattr(last, "type", None) != "ai":
                    continue
                content = extract_text(last.content)
                if not content or not content.startswith(sent):
                    continue
                delta = content[len(sent) :]
                if delta:
                    sent = content
                    yield event({"delta": delta})
            yield event({"done": True})
        except Exception as exc:  # 流式接口要把异常也推给前端，否则前端一直转圈
            yield event({"error": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
