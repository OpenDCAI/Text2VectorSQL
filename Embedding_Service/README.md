# Embedding Service 模块说明

本模块提供一个高性能、支持多模型、多GPU的文本和图像向量化API服务。它基于FastAPI和Sentence-Transformers构建,能够自动管理模型下载与缓存。

## 主要功能

- **多模型支持**: 可通过`config.yaml`配置文件同时加载和管理多个不同的向量化模型。
- **高性能**:
    - 基于FastAPI和Uvicorn,提供异步处理能力。
    - 支持通过`tensor_parallel_size`配置为单个模型启动多进程池,充分利用多GPU资源进行张量并行计算。
- **自动模型缓存**:
    - 服务启动时,会自动检查`config.yaml`中指定的本地模型路径。
    - 如果模型不存在,服务会从Hugging Face Hub自动下载并保存到指定路径,后续启动将直接从本地加载,避免重复下载。
- **统一的API接口**:
    - `/embed`: 核心接口,接收文本或图像数据,返回对应的向量表示。支持文本和图像两种输入模式。
    - `/health`: 健康检查接口,返回服务运行状态和已加载的模型列表。
- **客户端示例**: 提供`multi_client.py`作为示例,演示如何请求API来获取文本和图像的向量。

## 文件结构

- `server.py`: 核心服务文件。实现了FastAPI应用,负责模型加载、多进程池管理和API请求处理。
- `multi_server.py`: `server.py`的多模态版本，支持同时处理文本和图像嵌入请求（弃用）。
- `multi_client.py`: 用于测试服务的客户端示例代码。
- `run.sh`: 启动服务的便捷脚本。
- `config.yaml`(需自行创建): 服务和模型的配置文件。

## 环境依赖

所有依赖项都已在`requirements.txt`中列出。

## 快速开始

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **创建配置文件**:
    在`Embedding_Service`目录下创建一个名为`config.yaml`的文件,并参考以下示例填入内容。

    ```yaml
    # config.yaml 示例
    server:
      host: "0.0.0.0"
      port: 8000
    
    models:
      # 文本模型示例
      - name: "all-MiniLM-L6-v2"
        hf_model_path: "sentence-transformers/all-MiniLM-L6-v2"
        local_model_path: "./models/all-MiniLM-L6-v2" # 本地缓存路径
        trust_remote_code: true
        max_model_len: 512
    
      # 多模态模型示例 (CLIP)
      - name: "clip-ViT-B-32"
        hf_model_path: "sentence-transformers/clip-ViT-B-32"
        local_model_path: "./models/clip-ViT-B-32"
        trust_remote_code: true
    
      # 多GPU张量并行示例 (需要至少2个GPU)
      # - name: "gemma-2b"
      #   hf_model_path: "google/gemma-2b"
      #   local_model_path: "./models/gemma-2b"
      #   trust_remote_code: true
      #   tensor_parallel_size: 2 # 使用2个GPU
    ```
    **注意**: 请确保`local_model_path`指向的目录存在或有权限创建。

3.  **启动服务**:
    ```bash
    bash run.sh
    ```
    服务启动后,会首先检查并准备模型。首次运行会因下载模型而耗时较长。

4.  **测试API**:
    - **健康检查**:
      ```bash
      curl http://localhost:8000/health
      ```
    - **获取文本向量**:
      ```bash
      curl -X POST http://localhost:8000/embed \
      -H "Content-Type: application/json" \
      -d 
      {
            "model": "all-MiniLM-L6-v2",
            "texts": [
              "Hello World!",
              "Machine Learning"
            ]
          }
      ```
    - **图片嵌入测试**:
      ```bash
      python multi_client.py
      ```
