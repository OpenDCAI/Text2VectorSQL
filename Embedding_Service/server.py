# embedding_server.py

import argparse
import logging
from contextlib import asynccontextmanager
from typing import List

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# 【修改】移除 vllm，导入 SentenceTransformer
from sentence_transformers import SentenceTransformer

# --- Globals ---
# 【修改】将 LLM_ENGINES 重命名为 MODELS，以反映其内容
CONFIG = {}
MODELS = {}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmbeddingService")

# --- Pydantic Models for API validation (保持不变) ---
class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="The name of the model to use for embedding (must match a name in config.yaml).")
    texts: List[str] = Field(..., description="A list of texts to embed.")

class EmbeddingResponse(BaseModel):
    model: str = Field(..., description="The name of the model used.")
    embeddings: List[List[float]] = Field(..., description="A list of embedding vectors.")

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads models on startup.
    """
    logger.info("Starting up Embedding Service...")
    
    # 加载配置的逻辑保持不变
    parser = argparse.ArgumentParser(description="Embedding Service with Sentence-Transformers and FastAPI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            CONFIG.update(config_data)
            logger.info(f"Configuration loaded from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}. Exiting.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}. Exiting.")
        exit(1)

    # 【修改】使用 SentenceTransformer 加载模型
    if not CONFIG.get('models'):
        logger.error("No models found in the configuration file. Exiting.")
        exit(1)
        
    for model_config in CONFIG['models']:
        model_name = model_config.get('name')
        hf_path = model_config.get('hf_model_path')
        if not model_name or not hf_path:
            logger.warning(f"Skipping invalid model configuration: {model_config}")
            continue
        
        logger.info(f"Loading model '{model_name}' from '{hf_path}' using Sentence-Transformers...")
        try:
            # 【核心修改】使用 SentenceTransformer 替换 vllm.LLM
            # SentenceTransformer 也支持 trust_remote_code 参数
            model = SentenceTransformer(
                model_name_or_path=hf_path,
                trust_remote_code=model_config.get('trust_remote_code', True)
            )
            MODELS[model_name] = model
            logger.info(f"Successfully loaded model '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")

    if not MODELS:
        logger.error("No models were successfully loaded. Shutting down.")
        exit(1)
        
    yield
    
    # --- Shutdown logic (if any) ---
    logger.info("Shutting down Embedding Service...")
    MODELS.clear()


# --- FastAPI App Initialization ---
app = FastAPI(
    # 【修改】更新服务描述
    title="Text2VectorSQL Embedding Service (Sentence-Transformers Backend)",
    description="A high-performance embedding API service powered by Sentence-Transformers and FastAPI.",
    version="1.0.1",
    lifespan=lifespan
)

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running."""
    # 【修改】更新对全局变量的引用
    return {"status": "ok", "loaded_models": list(MODELS.keys())}

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Takes a list of texts and returns their embeddings using the specified model.
    """
    # 【修改】从 MODELS 字典中获取模型
    if request.model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found. Available models: {list(MODELS.keys())}")
    
    model_engine = MODELS[request.model]
    
    try:
        # 【核心修改】调用 SentenceTransformer 的 encode 方法
        # convert_to_numpy=False 直接返回 list[list[float]]，符合 API 响应模型
        embeddings = model_engine.encode(request.texts, convert_to_numpy=False)
        
        return EmbeddingResponse(model=request.model, embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error during embedding process for model '{request.model}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during embedding.")

# --- Main execution block (保持不变) ---
if __name__ == "__main__":
    if not CONFIG:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="config.yaml")
        args, _ = parser.parse_known_args()
        try:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
                server_config = config_data.get('server', {})
                host = server_config.get('host', '0.0.0.0')
                port = server_config.get('port', 8000)
        except Exception:
            host, port = "0.0.0.0", 8000
    else:
        server_config = CONFIG.get('server', {})
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 8000)

    uvicorn.run(app, host=host, port=port)
