"""
Agent 层：赛博算命师
把两路混合检索（心理学 / 易经）封装成一个工具，交给 create_agent 调用。
这是 LangChain 1.0 推荐写法：retriever -> @tool -> create_agent。

关键设计：
- 每路独立判断"是否命中"，未命中就让模型用通用知识作答（兜底），不硬塞弱证据
- 易经命中时，把可用卦象写成硬约束，防止模型凭记忆编造导致"依据与回答对不上"
"""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from ingest import build_retrievers

load_dotenv()

# 服务启动时构建一次，两个知识库各一个混合检索器
retrievers = build_retrievers()


@tool
def search_oracles(question: str) -> str:
    """检索赛博算命知识库。当用户描述失眠、焦虑、压力、心慌、感情不顺等困扰时必须调用。
    会同时从心理学、易经两个角度检索相关条目；某角度无匹配时会明确告知。"""
    sections: list[str] = []
    gua_names: list[str] = []       # 本次实际命中的卦象，供末尾硬约束使用
    missed: list[str] = []          # 未命中的角度，交给模型用通用知识兜底

    for category, retriever in retrievers.items():
        docs, hit = retriever.retrieve_with_diagnostics(question)
        if not hit or not docs:
            missed.append(category)
            sections.append(f"【{category}】知识库中未检索到相关条目。")
            continue

        blocks = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.replace("#", "").strip()
            blocks.append(f"{i}. {content}")
            if category == "易经":
                m = re.match(r"^([\u4e00-\u9fff]+卦)", content)
                if m:
                    gua_names.append(m.group(1))
        sections.append(f"【{category}】\n" + "\n".join(blocks))

    parts = "\n\n".join(sections)

    # 兜底指令：未命中的角度让模型基于通用知识简要作答，避免硬编证据
    if missed:
        parts += (
            "\n\n【兜底说明】以上标注\"未检索到相关条目\"的角度，"
            f"即（{'、'.join(missed)}），知识库没有匹配内容。"
            "该部分请基于你自己的通用知识简要作答，并温和说明\"知识库暂无对应条目，以下为通用解读\"，"
            "不要声称这些内容来自知识库。"
        )

    # 硬约束：只在易经真正命中时生效，防止凭记忆编卦象导致与卜卦依据对不上。
    # 措辞必须强——"上面就是全部证据"这句不能省，它明确划定了证据边界，模型才不敢自行补充。
    if gua_names:
        parts += (
            "\n\n【硬性约束 · 最高优先级】上面【易经】段列出的就是本次可用的全部卦象，"
            "用户在界面上能看到这份依据，必须逐字对应：\n"
            f"- 允许引用的卦象，仅限：{'、'.join(gua_names)}。\n"
            "- 严禁写出上述清单之外的任何卦象名（哪怕你记得《易经》里有），"
            "也严禁引用证据外的卦辞。违反会与用户看到的依据矛盾。"
        )
    elif "易经" in missed:
        parts += (
            "\n\n【硬性约束 · 最高优先级】本次易经知识库未命中，不要编造任何具体卦象名或卦辞；"
            "用\"未得明确卦象\"等模糊说法带过，把重心放在心理学视角与建议上。"
        )
    return parts


SYSTEM_PROMPT = """你是"赛博算命师·玄机"，一位融合东方玄学与现代心理学的知识库助手。
用户带着困扰来找你（失眠、焦虑、压力大、心慌、感情不顺等）。

回答规则：
1. 必须先调用 search_oracles 工具检索两路知识，不要凭空编造。
2. 按以下结构作答，两部分都要有，每部分 2-4 句：
   【心理学视角】用心理学知识解释这种情绪/睡眠状态的机制
   【易经指引】只引用工具返回里已命中的卦象，说明其含义与处境的对应
3. 最后给一段温和的"玄机建议"，结合两路内容，给出可操作的安抚或行动建议。
4. 若工具标注某角度"未检索到相关条目"，按工具里的兜底说明处理：用通用知识作答并如实说明，不冒充知识库内容。
5. 语气神秘但不吓人，不说绝对凶吉，不替代医疗建议；严重失眠或情绪问题要建议寻求专业帮助。
6. 中文作答，简洁清晰，不要输出与问题无关的内容。
7. 铁律（最高优先级）：【易经指引】里出现的每一个卦象名，都必须是工具【易经】段里列出的卦象，
   一个都不能多、不能换。用户能在界面上看到检索依据，哪怕多写一个你记得的卦象，都会立刻穿帮。"""


def build_agent():
    """返回 create_agent 构建的智能体；未配置 Key 时返回 None，便于前端给出友好提示。"""
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-在这里"):
        print("[agent] 未配置 LLM_API_KEY，Agent 不可用")
        return None

    model_name = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash").strip()
    print(f"[agent] 问答模型: {model_name}")
    model = init_chat_model(
        model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
        temperature=0.7,
    )

    return create_agent(
        model=model,
        tools=[search_oracles],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),  # 多轮记忆：同一 thread_id 自动带历史
    )
