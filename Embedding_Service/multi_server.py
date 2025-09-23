# embedding_server.py

import argparse
import base64 # 【新增】导入 base64 用于解码
import io     # 【新增】导入 io 用于处理二进制数据
import logging
from contextlib import asynccontextmanager
from typing import List, Optional # 【修改】导入 Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image # 【新增】导入 Pillow 用于图像处理
# 导入 SentenceTransformer 保持不变
from sentence_transformers import SentenceTransformer

# --- Globals (保持不变) ---
CONFIG = {}
MODELS = {}

# --- Logging Setup (保持不变) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmbeddingService")

# --- Pydantic Models for API validation ---
class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="The name of the model to use for embedding (must match a name in config.yaml).")
    # 【修改】将 'texts' 和 'images' 设为可选字段，以支持不同类型的输入
    texts: Optional[List[str]] = Field(None, description="A list of texts to embed.")
    images: Optional[List[str]] = Field(None, description="A list of Base64-encoded images to embed.")

class EmbeddingResponse(BaseModel):
    model: str = Field(..., description="The name of the model used.")
    embeddings: List[List[float]] = Field(..., description="A list of embedding vectors.")

# --- FastAPI Lifespan Management (保持不变) ---
# The model loading logic does not need to change, as SentenceTransformer
# handles loading of both text and multi-modal models transparently.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads models on startup.
    """
    logger.info("Starting up Embedding Service...")
    
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
    
    logger.info("Shutting down Embedding Service...")
    MODELS.clear()


# --- FastAPI App Initialization (保持不变) ---
app = FastAPI(
    title="Text2VectorSQL Embedding Service (Sentence-Transformers Backend)",
    description="A high-performance API service for text and image embeddings, powered by Sentence-Transformers.",
    version="1.1.0", # Version bump
    lifespan=lifespan
)

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running."""
    return {"status": "ok", "loaded_models": list(MODELS.keys())}

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Takes a list of texts OR a list of images and returns their embeddings.
    """
    if request.model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found. Available models: {list(MODELS.keys())}")
    
    # 【核心修改】增加输入验证和处理逻辑
    # 1. 验证输入：确保只提供了 'texts' 或 'images' 中的一个
    if (request.texts is None and request.images is None) or \
       (request.texts is not None and request.images is not None):
        raise HTTPException(
            status_code=400, 
            detail="You must provide either 'texts' or 'images', but not both."
        )

    model_engine = MODELS[request.model]
    
    try:
        inputs_to_encode = []
        # 2. 根据输入类型准备数据
        if request.texts:
            inputs_to_encode = request.texts
            logger.info(f"Processing {len(request.texts)} texts with model '{request.model}'.")
        
        elif request.images:
            logger.info(f"Processing {len(request.images)} images with model '{request.model}'.")
            # 将 Base64 字符串解码为 Pillow 图像对象
            pil_images = []
            for b64_string in request.images:
                try:
                    # 从 Base64 字符串解码为字节
                    image_bytes = base64.b64decode(b64_string)
                    # 从字节数据创建 PIL.Image 对象
                    image = Image.open(io.BytesIO(image_bytes))
                    pil_images.append(image)
                except Exception as img_e:
                    logger.error(f"Failed to decode or open image: {img_e}")
                    raise HTTPException(status_code=400, detail="Invalid Base64-encoded image data provided.")
            inputs_to_encode = pil_images

        # 3. 调用 SentenceTransformer 的 encode 方法进行编码
        #    该方法可以透明地处理文本列表或图像对象列表
        embeddings = model_engine.encode(inputs_to_encode, convert_to_numpy=False)
        
        return EmbeddingResponse(model=request.model, embeddings=embeddings)
        
    except Exception as e:
        logger.error(f"Error during embedding process for model '{request.model}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during embedding: {e}")

# --- Main execution block (保持不变) ---
if __name__ == "__main__":
    # This part remains the same to parse config for host/port
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