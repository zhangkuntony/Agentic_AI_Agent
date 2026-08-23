# CorpDocs Chat Bot (Streamlit)

基于 LangGraph + RAG 的企业文档问答机器人，前端使用 Streamlit 构建。

## 功能概览

- **文档上传**：支持 PDF / TXT / EPUB / DOCX / DOC，上传后自动切分并向量化。
- **RAG 问答**：基于上传的文档进行检索增强生成。
- **工作流**：文档生成 → 合规检查 → 人工反馈（可选）→ 定稿 → 附加引用。

## 目录结构

```
chat_bot/
├── __init__.py
├── document_loader.py   # 文档加载器（按扩展名选择解析器）
├── llms.py              # LLM 与 Embedding 模型配置（火山引擎 Ark）
├── rag.py               # LangGraph 工作流（检索/生成/检查/定稿）
├── retriever.py         # 文档切分、向量化与检索
├── streamlit_app.py     # Streamlit 前端入口
└── README.md
```

## 环境要求

- Python 3.10+
- 安装项目根目录 `requirements.txt` 中的依赖（含 `streamlit`）

```powershell
cd e:\AI\Github\Agentic_AI_Agent
pip install -r requirements.txt
```

## 配置 API（必须）

在**项目根目录**配置 `.env` 文件（可复制 `.env.example` 修改）：

```ini
BASE_URL=your_api_base_url        # 火山引擎 Ark API 地址
API_KEY=your_api_key              # 火山引擎 API Key
MODEL_NAME=your_model_name        # 对话模型名，如 doubao-seed-1-6-250615
EMBEDDING_MODEL_NAME=your_embedding_model  # Embedding 模型名
```

> `.env` 由 `src/config/config.py` 通过 `load_dotenv()` 从**项目根目录**加载，因此 `.env` 必须放在 `e:\AI\Github\Agentic_AI_Agent\.env`，不能放在 `chat_bot` 目录下。

## 启动方式（正确姿势）

**重要：必须满足以下两个前提，否则会报 `ModuleNotFoundError` 或无法解析文档。**

### 前提 1：工作目录必须是 `src`（或等价地设置 PYTHONPATH）

代码里的导入路径是顶层包形式：

```python
from chapter04.chat_bot.document_loader import DocumentLoader
from config.config import API_KEY, BASE_URL, MODEL_NAME
```

`config`、`utils`、`chapter04` 都以 `src` 为根目录。但 **Streamlit 只把"主脚本所在目录"（`chat_bot/`）加入 `sys.path`，不会加入当前工作目录**，所以单纯 `cd src` 再 `streamlit run` 也可能报 `No module named 'chapter04'`。

任选一种方式解决：

- **方式 A（推荐）：设置 `PYTHONPATH`**

  ```powershell
  cd e:\AI\Github\Agentic_AI_Agent\src
  $env:PYTHONPATH = "e:\AI\Github\Agentic_AI_Agent\src"
  streamlit run chapter04/chat_bot/streamlit_app.py
  ```

- **方式 B：用 `python -m streamlit` 启动**（会把当前工作目录自动加入 `sys.path`）

  ```powershell
  cd e:\AI\Github\Agentic_AI_Agent\src
  python -m streamlit run chapter04/chat_bot/streamlit_app.py
  ```

- **方式 C：通过命令行直接指定模块路径 + PYTHONPATH 一次性设置**

  ```powershell
  cd e:\AI\Github\Agentic_AI_Agent\src
  set PYTHONPATH=e:\AI\Github\Agentic_AI_Agent\src && streamlit run chapter04/chat_bot/streamlit_app.py
  ```

### 前提 2：不要从 `chat_bot` 目录或项目根目录启动

- 从**项目根目录**启动：`streamlit run src/chapter04/chat_bot/streamlit_app.py` 会报 `No module named 'chapter04'`。
- `llms.py` 用了相对路径 `LocalFileStore("./cache/")` 缓存向量，必须在 `src` 下运行，缓存才会稳定落在 `src/cache/`。

### 启动命令汇总

```powershell
# 方式 A（推荐）
cd e:\AI\Github\Agentic_AI_Agent\src
$env:PYTHONPATH = "e:\AI\Github\Agentic_AI_Agent\src"
streamlit run chapter04/chat_bot/streamlit_app.py

# 可选参数
# --server.port 8600  指定端口
# --server.headless true  不自动打开浏览器
```

启动成功后终端会打印本地地址，浏览器访问：

```
Local URL:  http://localhost:8501
```

## 使用说明

1. 打开页面后，在右侧 **Document Management** 上传一个或多个文档。
2. 上传后脚本会立即切分并调用 Embedding 模型向量化（可在终端看到相关日志）。
3. 在左侧 **Chat Interface** 输入问题，机器人会基于上传文档检索并生成带引用的回答。

## 常见问题（FAQ）

### 1. 报 `ModuleNotFoundError: No module named 'chapter04'`

工作目录不对或未设置 `PYTHONPATH`。参见上文【前提 1】。

### 2. 上传 PDF 后终端只看到 `Skipping data after last boundary`，没有解析日志

该警告是 HTTP multipart 解析的无害信息，表示文件已正常上传。真正的解析日志来自 `load_document`（`logging.info(docs)`）和 `retriever.add_uploaded_docs`（`print(f"Failed to load ...")`）。如果没有这些日志，通常是**文件对象被存进 `session_state` 后失效**导致静默跳过——Streamlit 的 `UploadedFile` 会在脚本 rerun 后失效，必须在 `file_uploader` 返回的同一 rerun 内立即调用 `getvalue()` 读取为 bytes。

### 3. 对话时报 `ValueError: too many values to unpack`

`retriever.add_uploaded_docs` 现在接收 `[(name, bytes), ...]` 元组列表。请确认 `streamlit_app.py` 中：
- 已删除旧的 `docs = retriever.add_uploaded_docs(st.session_state.uploaded_files)`（会拿文件名列表去解包而报错）；
- 只在 `file_uploader` 返回处用 `[(file.name, file.getvalue())]` 调用。

### 4. 对话时报 `IndexError: list index out of range`

`graph.invoke` 传入的输入键必须是 `"messages"`（对应 `State.messages`），不能是 `"message"`：

```python
response = graph.invoke({"messages": HumanMessage(message)}, config=config)
```

### 5. Embedding 结果被缓存

`llms.py` 使用 `CacheBackedEmbeddings` + `LocalFileStore("./cache/")`，相同文本会命中缓存，避免重复计费。如需清空缓存，删除 `src/cache/` 目录即可。

## 技术栈

- [Streamlit](https://streamlit.io/) - 前端界面
- [LangChain](https://www.langchain.com/) - 文档加载、切分、Embedding
- [LangGraph](https://www.langchain.com/langgraph) - 状态化工作流编排
- [FAISS](https://github.com/facebookresearch/faiss) / `InMemoryVectorStore` - 向量存储与检索
- 火山引擎 Ark - 对话与 Embedding 模型服务
