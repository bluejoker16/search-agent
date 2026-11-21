# LangChain 1.0 RAG 智能问答助手（面试项目）

一个**能在 3 分钟内讲清楚**的企业知识库问答项目：Flask 式的极简分层，用 LangChain 1.0 新写法（`create_agent` + `@tool` + `init_chat_model`）串起完整 RAG 链路，FastAPI 提供流式接口，原生 HTML 做前端，**不引入 Vue/React，不搞微服务**。

> 面向面试：代码量小（3 个 Python 文件 + 1 个 HTML），每个文件职责单一，可以随时打开给面试官看。README 末尾准备了 12 个高频追问及回答要点。

---

## 一、技术栈

| 层 | 技术 | 版本 |
|---|---|---|
| 大模型 | DeepSeek-V4-Flash（经硅基流动，OpenAI 兼容协议） | deepseek-ai/DeepSeek-V4-Flash |
| Agent 编排 | LangChain `create_agent` | 1.x |
| 模型初始化 | `init_chat_model` | 1.x |
| 检索工具化 | `@tool` + `InMemoryVectorStore` | langchain-core 1.x |
| 文档切块 | `RecursiveCharacterTextSplitter` | langchain-text-splitters |
| Web 框架 | FastAPI + uvicorn | 0.110+ |
| 前端 | 原生 HTML / CSS / JS + SSE | 零构建 |

**Embedding 双模式**（这是项目的设计亮点）：
- 不配 Key：本地字符 bigram 哈希向量，**零成本跑通全链路**
- 配了 Key（硅基流动，问答和向量化共用一个）：问答用 `DeepSeek-V4-Flash`，向量用免费 BGE-M3，**真实语义检索**

---

## 二、快速开始

```bash
# 1. 安装依赖
cd langchain-rag-interview
pip install -r requirements.txt

# 2. 配置 Key（硅基流动一个 Key 同时用于问答和向量化）
cp .env.example .env
#    编辑 .env，把 LLM_API_KEY 换成你的硅基流动 Key
#    地址：https://cloud.siliconflow.cn → 账户管理 → API 密钥
#    Embedding 的 Key 留空即可，会自动复用同一个 Key（BGE-M3 免费）

# 3. 启动
uvicorn app:app --reload --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

`.env` 里 `LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash` 是你指定的问答模型（支持工具调用、1M 上下文）。`EMBEDDING_API_KEY` 留空即可，代码自动复用 `LLM_API_KEY`，BGE-M3 在硅基流动免费。想换模型只改 `LLM_MODEL` 和 `LLM_BASE_URL`，代码不用动。

验证链路（不调用大模型，纯本地）：
```bash
python ingest.py
```

---

## 三、项目结构

```
langchain-rag-interview/
├── app.py                  # 接口层：FastAPI 路由、SSE 流式输出
├── agent.py                # Agent 层：@tool 检索工具 + create_agent 编排
├── ingest.py               # 数据层：加载 -> 切块 -> 向量化 -> 检索器
├── static/index.html       # 前端：原生聊天界面 + 流式渲染
├── data/docs/              # 知识库文档（放 .md / .txt 即可被索引）
│   └── 星云客服云平台_产品手册.md
├── requirements.txt
├── .env.example
└── README.md
```

**分层讲解口径**（面试时按这个顺序讲，30 秒）：

> 三层结构，从上到下是 `app.py` 接口层、`agent.py` 编排层、`ingest.py` 数据层。接口层只管 HTTP 和流式推送；编排层把检索包装成一个工具交给 `create_agent`，让模型自己决定要不要查知识库；数据层负责把文档变成可检索的向量库。前端是纯原生 HTML，通过 SSE 接收流式回答。

---

## 四、RAG 链路怎么跑的

```
用户提问
   │
   ▼
FastAPI /api/chat ──► create_agent（带对话记忆）
                          │
                          ├─ 模型判断：需要查内部资料？
                          │        │
                          │        ▼
                          │   @tool search_knowledge_base
                          │        │
                          │        ▼
                          │   retriever 检索 Top-3 片段
                          │        │
                          │        ▼
                          │   片段作为上下文塞回 Prompt
                          │
                          ▼
                    模型基于上下文生成答案
                          │
                          ▼
                    SSE 流式推给前端 + 展示检索证据
```

**索引阶段**（服务启动时执行一次）：`data/docs/*.md` → 按空行分段 → `RecursiveCharacterTextSplitter`（300 字/50 重叠）→ Embedding → `InMemoryVectorStore`。

---

## 五、 LangChain 1.0 的关键写法（面试重点）

你学完教程后，这个项目用的就是 1.0 新 API，和网上 0.x 教程完全不同：

| 能力 | 0.x 老写法（网上大部分项目） | 本项目（1.0） |
|---|---|---|
| 模型初始化 | `ChatOpenAI(...)` 直接实例化 | `init_chat_model(..., model_provider="openai")` 统一工厂 |
| Agent | `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| Prompt | `prompt=` 参数 | `system_prompt=` 参数 |
| 记忆 | `ConversationBufferMemory` | `checkpointer=InMemorySaver()` + `thread_id` |
| 检索接入 | `RetrievalQA` / `ConversationalRetrievalChain` | 检索包成 `@tool`，交给 Agent 调用 |
| Embedding | `langchain_community.embeddings.XXX` | `langchain-core` 的 `Embeddings` 子类 / `langchain_openai` |

**一句话记忆法**：`model_provider` 决定请求协议，`base_url` 决定请求地址，`model` 决定实际模型，`api_key` 决定凭证。换厂商只改这四个。

---

## 六、面试高频问题 & 回答要点

**1. 讲一下这个项目？**
> 一句话：企业知识库问答。前端提问 → FastAPI → LangChain 1.0 Agent → 调用检索工具拿文档片段 → DeepSeek 基于片段作答 → SSE 流式返回。三个 Python 文件分别对应数据层、编排层、接口层。

**2. RAG 是什么，为什么要用？**
> 大模型不知道企业私有数据，直接微调成本高、更新慢。RAG 在生成前先检索相关资料塞进上下文，让模型"开卷考试"。优点是知识更新只需改文档、可溯源（本项目前端能看检索片段）、成本低。

**3. 文档为什么要切块？chunk_size 怎么定？**
> 两个原因：一是模型上下文窗口有限；二是整篇文档向量化会稀释语义，检索精度下降。项目用 `RecursiveCharacterTextSplitter`，300 字一块、50 字重叠，重叠是为了避免一个完整语义被切断。实际要按文档类型调：FAQ 可以小一些，论文可以大一些。

**4. 为什么把检索包成 `@tool` 给 Agent，而不是直接 LCEL 串 `retriever | prompt | model`？**
> LCEL 链是"每次必查"，简单直接但不够灵活。包成工具后，模型自己判断要不要查：问"公司产品功能"会调用检索，问"帮我写个 for 循环"就直接用通用知识回答。代价是引入一次工具调用的延迟和不确定性，所以我在 system prompt 里明确规定了何时必须调用。

**5. LangChain 1.0 相比 0.x 改了什么？**
> 主包收敛到 Agent 核心能力；老链和部分 Retriever 移到 `langchain-classic`；`create_react_agent` 被 `create_agent` 取代；提示词参数改名 `system_prompt`；记忆改用 checkpointer；流式节点名从 `agent` 变成 `model`。

**6. DeepSeek 没有 embedding 接口，你怎么解决的？**
> 做了双模式降级。默认用本地字符 bigram 哈希向量，零成本把"加载→切块→向量化→检索→生成"整条链路跑通；配了 SiliconFlow Key 就用 BGE-M3 语义向量。BGE 查询侧还要加指令前缀，文档侧不加，这是训练方式决定的。

**7. 为什么用内存向量库，不用 Chroma / Milvus？**
> 项目定位是面试演示，文档量只有几十 KB，内存库启动即建、零外部服务，面试官clone下来不用装数据库就能跑。真实项目我会换 Chroma 或 pgvector 做持久化，并加增量索引，避免每次重启全量重建。

**8. 多轮对话怎么实现的？**
> `create_agent` 配 `InMemorySaver` 作 checkpointer，前端每次带同一个 `thread_id`，历史消息由框架自动注入。注意生产环境要换成数据库持久化，否则服务重启记忆就丢了。

**9. 怎么防止模型胡编？**
> 三层约束：system prompt 要求"只依据检索内容作答，检索不到就明说不知道"；检索结果带来源文件名，模型要输出"参考来源"；前端展示原始检索片段，人工可核对。

**10. 流式输出怎么做的？**
> 后端 `/api/chat` 返回 `StreamingResponse`，用 `agent.stream(..., stream_mode="values")` 逐步拿 AI 消息，做增量 diff 后按 SSE `data:` 格式推送；前端用 `ReadableStream` 逐块解析并追加到气泡里。异常也通过 SSE 推 `error` 事件，避免前端一直转圈。

**11. 检索不到相关内容怎么办？**
> 工具返回固定提示"知识库中没有检索到相关内容"，system prompt 要求模型如实告知而不是编造。进一步优化可以设相似度阈值、做 query 改写或多路召回。

**12. 项目有什么不足，怎么改进？**
> 三点：一是内存向量库不持久化；二是没有 rerank，Top-3 直接进 prompt，可以接 BGE-reranker 重排；三是没有做 query 改写和混合检索，后续可以加 BM25 + 向量混合召回。

---

## 七、可扩展方向（面试官问"还能怎么改"时说）

- 换持久化向量库（Chroma / pgvector），支持文档增删改
- 加 reranker 模型做二次排序，提升 Top-K 质量
- Query 改写 + 多路召回（BM25 + 向量）
- 上传文档接口，前端做文件管理
- 接 LangSmith 做链路追踪，统计检索命中率
- 加相似度阈值，低于阈值直接拒答

---

MIT License · 学习交流用
