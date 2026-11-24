"""
FastAPI 接口层（赛博算命）：
  GET  /                 聊天页面
  GET  /api/health       健康检查
  POST /api/chat         SSE 流式问答（核心接口）
  POST /api/retrieve     两路检索证据（独立调试用；前端展示已改用 /api/chat 回传的同源证据）
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agent import build_agent, retrievers

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="赛博算命 · 心理 × 易经")

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
    """返回两路检索证据 —— 面试演示时打开它，直观证明答案来自两个知识库。"""
    groups = []
    for category, retriever in retrievers.items():
        docs, hit = retriever.retrieve_with_diagnostics(req.question)
        groups.append({
            "category": category,
            "hit": hit,
            "docs": [
                {"source": d.metadata.get("source", "未知来源"), "content": d.page_content}
                for d in docs
            ],
        })
    return {"groups": groups}


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


def parse_tool_evidence(raw: str) -> list[dict]:
    """把工具返回的证据文本解析成前端展示用的 groups。

    工具输出形如：
        【心理学】
        1. 考试焦虑与表现焦虑
        ...
        【易经】
        1. 困卦（...）
        ...
        【硬性约束】...
    """
    import re

    groups: list[dict] = []
    # 按【类别】切分，类别名即知识库名
    parts = re.split(r"【(心理学|易经)】", raw)
    # parts = [前导, 类别1, 内容1, 类别2, 内容2, ...]
    for i in range(1, len(parts) - 1, 2):
        category, body = parts[i], parts[i + 1]
        # 去掉末尾的【硬性约束】/【兜底说明】段，避免被当成证据
        body = re.split(r"【(?:硬性约束|兜底说明)】", body)[0]
        # 未命中：这一路知识库没有相关内容，标记出来让前端如实展示
        if "未检索到相关条目" in body:
            groups.append({"category": category, "hit": False, "docs": []})
            continue
        docs = []
        for block in re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.\s*|\Z)", body, re.S):
            content = block.strip()
            if content:
                docs.append({"source": f"{category}.md", "content": content})
        if docs:
            groups.append({"category": category, "hit": True, "docs": docs})
    return groups


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

        # 记录本轮开始前已有的 tool 消息 ID。多轮会话里历史工具消息会留在 state 中，
        # 只接受"本轮新产生"的 tool 消息，才能拿到本次真正用到的证据。
        try:
            prev_state = agent.get_state(config)
            seen_tool_ids = {
                m.id for m in prev_state.values.get("messages", [])
                if getattr(m, "type", None) == "tool" and getattr(m, "id", None)
            }
        except Exception:
            seen_tool_ids = set()

        sent = ""
        evidence_sent = False
        try:
            for state in agent.stream(
                {"messages": [{"role": "user", "content": req.question}]},
                config=config,
                stream_mode="values",
            ):
                messages = state.get("messages", [])
                if not messages:
                    continue

                # 把 Agent 本轮实际使用的工具证据推给前端，保证"卜卦依据"和回答同源。
                # 只认本轮新产生的 tool 消息（ID 不在开始前集合里），避免多轮串台。
                if not evidence_sent:
                    for msg in messages:
                        if getattr(msg, "type", None) != "tool":
                            continue
                        mid = getattr(msg, "id", None)
                        if mid and mid in seen_tool_ids:
                            continue
                        raw = extract_text(getattr(msg, "content", ""))
                        groups = parse_tool_evidence(raw)
                        if groups:
                            yield event({"evidence": groups})
                            evidence_sent = True
                            if mid:
                                seen_tool_ids.add(mid)  # 防止同一条消息在后续 state 里重复推送
                            break

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
