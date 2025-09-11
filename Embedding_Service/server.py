# embedding_server.py

import argparse
import logging
from contextlib import asynccontextmanager
from typing import List

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM

# --- Globals ---
# Will be populated by the config file and startup event
CONFIG = {}
LLM_ENGINES = {}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmbeddingService")

# --- Pydantic Models for API validation ---
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
    
    # Load configuration
    parser = argparse.ArgumentParser(description="Embedding Service with vLLM and FastAPI")
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

    # Load models into vLLM engines
    if not CONFIG.get('models'):
        logger.error("No models found in the configuration file. Exiting.")
        exit(1)
        
    for model_config in CONFIG['models']:
        model_name = model_config.get('name')
        hf_path = model_config.get('hf_model_path')
        if not model_name or not hf_path:
            logger.warning(f"Skipping invalid model configuration: {model_config}")
            continue
        
        logger.info(f"Loading model '{model_name}' from '{hf_path}'...")
        try:
            # Note: vLLM's LLM class is now the standard way for both generation and encoding
            llm = LLM(
                model=hf_path,
                trust_remote_code=model_config.get('trust_remote_code', True),
                tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
                max_model_len=model_config.get('max_model_len', 512),
                # For embedding, we don't need sampling, so engine is simpler
            )
            LLM_ENGINES[model_name] = llm
            logger.info(f"Successfully loaded model '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")

    if not LLM_ENGINES:
        logger.error("No models were successfully loaded. Shutting down.")
        exit(1)
        
    yield
    
    # --- Shutdown logic (if any) ---
    logger.info("Shutting down Embedding Service...")
    LLM_ENGINES.clear()


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Text2VectorSQL Embedding Service",
    description="A high-performance embedding API service powered by vLLM and FastAPI.",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running."""
    return {"status": "ok", "loaded_models": list(LLM_ENGINES.keys())}

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Takes a list of texts and returns their embeddings using the specified model.
    """
    if request.model not in LLM_ENGINES:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found. Available models: {list(LLM_ENGINES.keys())}")
    
    engine = LLM_ENGINES[request.model]
    
    try:
        # 【修正】直接调用同步的 encode 方法，移除 await
        request_outputs = engine.encode(request.texts)
        
        # 提取嵌入向量的逻辑保持不变
        embeddings = [output.outputs.embedding for output in request_outputs]
        
        return EmbeddingResponse(model=request.model, embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error during embedding process for model '{request.model}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during embedding.")

# --- Main execution block ---
if __name__ == "__main__":
    if not CONFIG:
        # This block will run if the script is started directly,
        # but lifespan event is the main logic driver.
        # We need to pre-load config to get host/port for uvicorn.run
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
        # Config loaded via lifespan
        server_config = CONFIG.get('server', {})
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 8000)

    uvicorn.run(app, host=host, port=port)