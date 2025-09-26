# server.py

import os
import argparse
import asyncio
import functools
import logging
from contextlib import asynccontextmanager
from threading import Lock
from typing import List, Dict, Any

import uvicorn
import yaml
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# --- Globals ---
CONFIG: Dict[str, Any] = {}
MODELS: Dict[str, Dict[str, Any]] = {}
# 新增一个线程锁，以防止在多worker模式下可能出现的下载竞争问题
model_download_lock = Lock()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmbeddingService")

# --- 【新增】模型下载与准备的辅助函数 ---
def prepare_model_path(model_config: Dict[str, Any]) -> str:
    """
    检查本地模型路径是否存在。如果不存在，则从Hugging Face下载并保存。
    返回最终可供加载的本地模型路径。
    此函数设计为线程安全。
    """
    hf_path = model_config.get("hf_model_path")
    local_path = model_config.get("local_model_path")

    if not hf_path or not local_path:
        raise ValueError(f"模型 '{model_config.get('name')}' 的配置缺少 'hf_model_path' 或 'local_model_path'。")

    # 使用一个关键文件（如config.json）来判断模型是否已完整存在
    local_config_file = os.path.join(local_path, "config.json")

    if os.path.exists(local_config_file):
        logger.info(f"在本地路径 '{local_path}' 找到模型。将直接加载。")
        return local_path
    
    # 如果本地不存在，则加锁以确保只有一个进程/线程执行下载
    with model_download_lock:
        # 双重检查，防止在等待锁的过程中，其他线程已经下载完毕
        if os.path.exists(local_config_file):
            logger.info(f"在等待锁后，发现模型已存在于 '{local_path}'。")
            return local_path

        logger.warning(f"本地模型未找到。开始从 '{hf_path}' 下载...")
        logger.warning("（首次下载会花费一些时间，请耐心等待...）")
        
        try:
            # 1. 下载模型到 huggingface 的默认缓存中
            model = SentenceTransformer(hf_path)
            # 2. 将完整的模型文件保存到我们指定的永久本地路径
            model.save(local_path)
            logger.info(f"✅ 模型成功下载并保存到: '{local_path}'")
            return local_path
        except Exception as e:
            logger.error(f"❌ 下载或保存模型 '{hf_path}' 时发生错误: {e}", exc_info=True)
            raise

# --- Pydantic Models (保持不变) ---
class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="The name of the model to use for embedding (must match a name in config.yaml).")
    texts: List[str] = Field(..., description="A list of texts to embed.")

class EmbeddingResponse(BaseModel):
    model: str = Field(..., description="The name of the model used.")
    embeddings: List[List[float]] = Field(..., description="A list of embedding vectors.")

# --- FastAPI Lifespan Management (已修改) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    处理启动和关闭事件。
    在启动时，会先准备好模型（下载或使用缓存），然后再加载到GPU。
    """
    global CONFIG, MODELS
    logger.info("Starting up Embedding Service...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args, _ = parser.parse_known_args()
    
    try:
        with open(args.config, 'r') as f:
            CONFIG.update(yaml.safe_load(f))
            logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}. Exiting.", exc_info=True)
        exit(1)

    if not CONFIG.get('models'):
        logger.error("配置文件中未找到模型定义. Exiting.")
        exit(1)
        
    for model_config in CONFIG['models']:
        model_name = model_config.get('name')
        try:
            # 【核心修改】在加载模型前，先调用辅助函数确保模型已在本地准备好
            final_model_path = prepare_model_path(model_config)

            logger.info(f"开始从本地路径 '{final_model_path}' 加载模型 '{model_name}'...")
            
            # 从准备好的本地路径加载模型
            model = SentenceTransformer(
                model_name_or_path=final_model_path,
                trust_remote_code=model_config.get('trust_remote_code', True)
            )
            
            max_len = model_config.get('max_model_len')
            if isinstance(max_len, int):
                model.max_seq_length = max_len
                logger.info(f"模型 '{model_name}' 的最大序列长度设置为 {max_len}.")

            pool = None
            parallel_size = model_config.get('tensor_parallel_size', 1)
            
            if parallel_size > 1:
                if not torch.cuda.is_available():
                    logger.warning(f"CUDA 不可用。模型 '{model_name}' 将在 CPU 单进程上运行。")
                else:
                    num_gpus = torch.cuda.device_count()
                    if num_gpus < parallel_size:
                        logger.warning(f"请求 {parallel_size} 个 GPU, 但只有 {num_gpus} 个可用。将使用所有可用的GPU。")
                        parallel_size = num_gpus

                    target_devices = [f'cuda:{i}' for i in range(parallel_size)]
                    logger.info(f"为模型 '{model_name}' 在设备 {target_devices} 上启动多进程池...")
                    pool = model.start_multi_process_pool(target_devices=target_devices)
                    logger.info(f"✅ 成功为 '{model_name}' 启动多进程池。")

            MODELS[model_name] = {"engine": model, "pool": pool}
            logger.info(f"✅ 模型 '{model_name}' 加载成功。")

        except Exception as e:
            logger.error(f"❌ 加载模型 '{model_name}' 失败: {e}", exc_info=True)

    if not MODELS:
        logger.error("没有任何模型被成功加载。服务将关闭。")
        exit(1)
        
    yield
    
    logger.info("Shutting down Embedding Service...")
    for model_name, model_data in MODELS.items():
        if model_data.get("pool"):
            logger.info(f"正在停止模型 '{model_name}' 的多进程池...")
            SentenceTransformer.stop_multi_process_pool(model_data["pool"])
    MODELS.clear()

# --- FastAPI App & Endpoints (保持不变) ---
app = FastAPI(
    title="Intelligent Embedding Service",
    description="自动处理模型缓存的高性能、多GPU嵌入服务。",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    model_status = {name: f"pool: {'active' if data.get('pool') else 'inactive'}" for name, data in MODELS.items()}
    return {"status": "ok", "loaded_models": model_status}

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    model_entry = MODELS.get(request.model)
    if not model_entry:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found. Available models: {list(MODELS.keys())}")
    
    model_engine = model_entry["engine"]
    pool = model_entry.get("pool")
    
    try:
        loop = asyncio.get_event_loop()
        encode_func = functools.partial(model_engine.encode, request.texts, pool=pool, batch_size=256, convert_to_numpy=False)
        embeddings = await loop.run_in_executor(None, encode_func)
        return EmbeddingResponse(model=request.model, embeddings=embeddings)
    except Exception as e:
        logger.error(f"处理嵌入请求时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during embedding.")
    
# --- Main execution block (保持不变) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    args, _ = parser.parse_known_args()

    host, port = "0.0.0.0", 8000
    try:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            server_config = config_data.get('server', {})
            host = server_config.get('host', host)
            port = server_config.get('port', port)
    except Exception:
        pass

    uvicorn.run("server:app", host=host, port=port)
