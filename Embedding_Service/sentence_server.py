import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
# 导入 models 模块
from sentence_transformers import SentenceTransformer, models
from typing import List

# --- 手动构建模型 ---
# model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
model_name = "sentence-transformers/clip-ViT-B-32"

# # 1. 首先，创建一个 Transformer 模块，这是对 HuggingFace 模型的封装
# word_embedding_model = models.Transformer(model_name)

# # 2. CLIP 模型本身就能输出句子级别的 embedding，所以不需要添加池化层。
# #    models.Transformer 模块足够智能，能直接输出我们想要的结果。

# # 3. 将这个模块传入 SentenceTransformer
# model = SentenceTransformer(modules=[word_embedding_model])
model = SentenceTransformer(model_name)

# --- 后续的 FastAPI 代码保持不变 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleEmbeddingService")

app = FastAPI(title="Simple Embedding Service")

class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="A list of texts to embed.")

class EmbeddingResponse(BaseModel):
    model: str
    embeddings: List[List[float]]

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    logger.info(f"Received {len(request.texts)} texts for embedding.")
    
    # 直接获取 list 结果，不再调用 .tolist()
    embeddings = model.encode(request.texts, convert_to_numpy=False)
    
    logger.info("Successfully created embeddings.")
    return EmbeddingResponse(model=model_name, embeddings=embeddings)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
