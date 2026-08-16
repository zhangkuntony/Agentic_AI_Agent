from config.config import API_KEY, BASE_URL, EMBEDDING_MODEL_NAME
from langchain_core.embeddings import Embeddings
from typing import List
from volcenginesdkarkruntime import Ark

class ArkMultimodalEmbeddings(Embeddings):
    """
    火山引擎多模态 embedding 模型的 LangChain 兼容封装。
    纯文本场景下，把文本包装成 {"type": "text", "text": ...} 结构调用。
    """

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL,
                 model: str = EMBEDDING_MODEL_NAME):
        self.client = Ark(api_key=api_key, base_url=base_url)
        self.model = model

    def _embed(self, inputs: List[dict]) -> List[List[float]]:
        """内部方法：逐条调用，因为该接口一次只返回一个融合向量"""
        embeddings = []
        for item in inputs:
            resp = self.client.multimodal_embeddings.create(
                model=self.model,
                encoding_format="float",
                input=[item],       # 每条单独调用，保证得到独立向量
            )
            embeddings.append(resp.data.embedding)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码文档，返回与 texts 一一对应的向量列表"""
        inputs = [{"type": "text", "text": t} for t in texts]
        return self._embed(inputs)

    def embed_query(self, text: str) -> List[float]:
        """编码单条查询文本"""
        return self.embed_documents([text])[0]