"""
Agent 层：把「检索」包装成工具，交给 create_agent 自己决定何时调用。
这是 LangChain 1.0 推荐写法：retriever -> @tool -> create_agent。
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from ingest import build_retriever

load_dotenv()

# 服务启动时构建一次检索器，避免每次提问都重新切块+向量化
retriever = build_retriever()


@tool
def search_knowledge_base(query: str) -> str:
    """检索公司内部知识库。当用户询问产品功能、价格、部署方式、售后政策等内部资料时必须调用此工具。"""
    docs = retriever.invoke(query)
    if not docs:
        return "知识库中没有检索到相关内容。"
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        blocks.append(f"【片段{i}｜来源：{source}】\n{doc.page_content}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = """你是"小知"，公司内部知识库智能问答助手。严格遵守：
1. 涉及产品、制度、流程等内部信息时，必须先调用 search_knowledge_base 工具检索，不要凭记忆回答。
2. 只依据检索到的内容作答。检索不到就明确说"知识库中没有相关信息"，不要编造。
3. 回答最后单独一行写：参考来源：<文档名>（多个用顿号分隔）。
4. 使用中文，分点回答，简洁清晰。
5. 与知识库无关的通用问题（如闲聊、写代码）可以正常回答，但不要虚构公司内部信息。"""


def build_agent():
    """返回 create_agent 构建的智能体；未配置 Key 时返回 None，便于前端给出友好提示。"""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-在这里"):
        print("[agent] 未配置 LLM_API_KEY，Agent 不可用")
        return None

    # 关键：model_provider="openai" 表示用 OpenAI 兼容协议，不是用 OpenAI 的模型。
    # 换模型只改 .env 的 LLM_MODEL / LLM_BASE_URL / LLM_API_KEY，代码不动。
    model_name = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash").strip()
    print(f"[agent] 问答模型: {model_name}")
    model = init_chat_model(
        model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
        temperature=0.3,
    )

    return create_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),  # 多轮记忆：同一 thread_id 自动带历史
    )
