import os
import re
import shutil
import sqlite3
import logging
from typing import Optional, List, Dict, Any
import torch
import itertools
import multiprocessing

# 确保已安装所需库: pip install Pillow sentence-transformers tqdm numpy
try:
    from PIL import Image, UnidentifiedImageError
    from sentence_transformers import SentenceTransformer, models
    from tqdm import tqdm
    import numpy as np
    import concurrent.futures
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you have installed the necessary libraries: pip install Pillow sentence-transformers tqdm numpy")
    exit(1)

import sqlite_vec

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ★★★ 新增：状态与缓存管理器 ★★★ ---
import json

class StateManager:
    """一个简单的类，用于管理脚本执行状态和缓存。"""
    def __init__(self, output_db_path: str):
        self.cache_dir = f"{output_db_path}.cache"
        self.state_file = os.path.join(self.cache_dir, "state.json")
        self.embeddings_file = os.path.join(self.cache_dir, "embeddings.npz")
        self.state = {}
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_state()

    def _load_state(self):
        """从文件加载状态。"""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                self.state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.state = {}

    def save_state(self):
        """将当前状态保存到文件。"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """获取一个状态值。"""
        return self.state.get(key, default)

    def set(self, key: str, value: Any):
        """设置一个状态值并立即保存。"""
        self.state[key] = value
        self.save_state()

    def reset(self):
        """重置状态，清空缓存目录。"""
        logging.info(f"正在重置状态并清空缓存目录: {self.cache_dir}")
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.state = {}

    def is_cache_valid(self, input_sql_path: str) -> bool:
        """检查缓存是否对当前输入文件有效。"""
        if not self.get('embeddings_generated'):
            return False
        
        stored_path = self.get('input_sql_path')
        stored_mtime = self.get('input_sql_mtime')
        
        if not stored_path or not stored_mtime:
            return False

        if stored_path != input_sql_path:
            logging.warning("输入SQL文件路径已更改，缓存失效。")
            return False
        
        try:
            current_mtime = os.path.getmtime(input_sql_path)
            if stored_mtime != current_mtime:
                logging.warning("输入SQL文件内容已更改，缓存失效。")
                return False
        except FileNotFoundError:
            logging.warning("找不到原始输入SQL文件，缓存失效。")
            return False
            
        return True
    
def load_embedding_model(model_name: str, cache_folder: str = None, device: str = 'cpu'):
    """
    智能加载 Sentence Transformer 模型。
    会首先尝试标准方法加载，如果失败并检测到是CLIP模型的常见错误，
    则切换到手动模块化方法。如果仍然失败，则抛出最终错误。

    :param model_name: Hugging Face 上的模型名称或本地路径
    :param cache_folder: 模型缓存的本地文件夹路径
    :param device: 运行模型的设备 ('cpu', 'cuda', etc.)
    :return: 加载好的 SentenceTransformer 模型对象
    """
    
    # --- 步骤 1: 尝试使用标准方法直接加载 ---
    try:
        logging.info(f"【尝试方法 1】使用标准方法加载模型: '{model_name}'")
        model = SentenceTransformer(
            model_name_or_path=model_name,
            device=device,
            cache_folder=cache_folder
        )
        logging.info("✅ 标准方法加载成功！")
        return model
    except AttributeError as e:
        # 捕捉到 AttributeError，检查是否是 CLIP 模型的典型错误
        if 'CLIPConfig' in str(e) and 'hidden_size' in str(e):
            logging.warning(f"⚠️ 标准方法失败，检测到CLIP模型结构不兼容错误。")
            logging.info(f"【尝试方法 2】切换到CLIP手动加载方法...")
            
            # --- 步骤 2: 尝试使用手动模块化方法加载 CLIP 模型 ---
            try:
                clip_model_module = models.CLIPModel(model_name=model_name)
                model = SentenceTransformer(
                    modules=[clip_model_module],
                    device=device,
                    cache_folder=cache_folder
                )
                logging.info("✅ CLIP手动加载方法成功！")
                return model
            except Exception as clip_error:
                logging.error(f"❌ CLIP手动加载方法也失败了。")
                # 如果手动方法也失败了，就将错误抛出
                raise clip_error
        else:
            # 如果是其他类型的 AttributeError，说明是未知问题，直接抛出
            logging.error(f"❌ 加载失败，发生未预料的 AttributeError。")
            raise e
    except Exception as other_error:
        # 捕捉其他所有可能的错误（如网络问题、模型不存在等）
        logging.error(f"❌ 加载失败，发生未知错误。")
        raise other_error
    
def _parse_sql_insert_line(line: str, table_name: str) -> Optional[Dict[str, Any]]:
    """(功能完整版) 解析单行INSERT语句，提取所有列和值。"""
    insert_prefix = f'INSERT INTO "{table_name}"'
    values_marker = ") VALUES ("
    if not line.strip().startswith(insert_prefix): return None
    try:
        columns_end_index = line.find(values_marker)
        if columns_end_index == -1: return None
        columns_str = line[line.find('(') + 1 : columns_end_index]
        values_str = line[columns_end_index + len(values_marker) : -1]
        if values_str.endswith(';'):
            values_str = values_str[:-1]

        column_names = [col.strip().strip('"`') for col in columns_str.split(',')]
        
        value_parts = re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", values_str)
        values = []
        for val in value_parts:
            val = val.strip()
            if val.startswith("'") and val.endswith("'"):
                values.append(val[1:-1].replace("''", "'"))
            elif val.lower() == 'null':
                values.append(None)
            else:
                try:
                    # 尝试将值解析为数字
                    if '.' in val: values.append(float(val))
                    else: values.append(int(val))
                except ValueError:
                    values.append(val)

        if len(column_names) == len(values):
            return dict(zip(column_names, values))
    except Exception: return None
    return None

def _parse_image_insert_info(line: str) -> Optional[Dict[str, Any]]:
    """(高效版) 仅从 "Images" 表的INSERT语句中提取 article_id 和 filename。"""
    pattern = re.compile(r"VALUES\s*\(\s*\d+\s*,\s*(\d+)\s*,\s*'((?:[^']|'')*)'", re.IGNORECASE)
    match = pattern.search(line)
    if match:
        article_id_str, filename_str = match.groups()
        filename = filename_str.replace("''", "'")
        return {'article_id': int(article_id_str), 'filename': filename}
    return None

# --- ★★★ 关键修复：使用智能类型格式化 ★★★ ---
def _format_sql_value(value: Any) -> str:
    """将Python值安全地格式化为SQL字面量字符串，能正确处理数字和文本。"""
    if value is None:
        return "NULL"
    # 如果是数字，直接转为字符串，不加引号
    if isinstance(value, (int, float)):
        return str(value)
    # 其他所有类型（包括嵌入向量的字符串表示）都当作文本处理
    s = str(value)
    escaped_s = s.replace("'", "''")
    return f"'{escaped_s}'"


def build_final_db_with_images(
    input_sql_path: str,
    output_db_path: str,
    target_vec_db_path: str,
    table_to_enhance: str,
    model_name: str,
    model_cache_dir: Optional[str] = 'models',
    article_table: str = "Articles",
    image_data_root: Optional[str] = None
):
    if not os.path.exists(input_sql_path):
        logging.error(f"输入SQL脚本文件不存在: {input_sql_path}")
        return

    temp_model = load_embedding_model(model_name,model_cache_dir)
    embedding_dim = temp_model.get_sentence_embedding_dimension()
    del temp_model
    logging.info(f"模型加载成功，嵌入维度: {embedding_dim}")

    # --- 2. 预加载文章标题 ---
    article_title_cache = {}
    try:
        logging.info(f"正在从 '{target_vec_db_path}' 预加载文章标题...")
        conn_lookup = sqlite3.connect(f'file:{target_vec_db_path}?mode=ro', uri=True)
        conn_lookup.enable_load_extension(True)
        sqlite_vec.load(conn_lookup)
        cursor_lookup = conn_lookup.cursor()
        cursor_lookup.execute(f'SELECT article_id, title FROM "{article_table}"')
        for article_id, title in cursor_lookup.fetchall():
            article_title_cache[article_id] = title
        conn_lookup.close()
        logging.info(f"成功预加载 {len(article_title_cache)} 个文章标题。")
    except Exception as e:
        logging.error(f"预加载文章标题失败: {e}", exc_info=True)

    # --- 3. 读取并分割SQL ---
    logging.info("正在读取并将SQL脚本分割成独立命令...")
    with open(input_sql_path, 'r', encoding='utf-8') as f:
        sql_script_content = f.read()
    sql_statements = [stmt.strip() for stmt in sql_script_content.split('-- VEC_SQL_SEPARATOR --') if stmt.strip()]
    
    # --- 4. 收集图片任务 ---
    logging.info("正在收集所有图片处理任务...")
    image_processing_tasks = []
    create_table_pattern = re.compile(r'CREATE\s+VIRTUAL\s+TABLE\s+"?' + re.escape(table_to_enhance) + r'"?', re.IGNORECASE)
    final_statements = [""] * len(sql_statements)
    
    insert_prefix = f'INSERT INTO "{table_to_enhance}"'
    for i, statement in tqdm(enumerate(sql_statements), total=len(sql_statements), desc="Collecting image tasks"):
        clean_statement = statement.rstrip(';')
        if clean_statement.strip().startswith(insert_prefix):
            info = _parse_image_insert_info(clean_statement)
            if info:
                image_path = None
                article_id, filename = info.get('article_id'), info.get('filename')
                if article_id is not None and filename is not None:
                    article_title = article_title_cache.get(article_id)
                    if article_title:
                        dir_name = re.sub(r'[\\/*?:"<>|]', "", article_title.replace(' ', '_'))
                        image_path = os.path.join(image_data_root, "image", dir_name, "img", filename)
                image_processing_tasks.append({'original_index': i, 'image_path': image_path})
            else:
                final_statements[i] = clean_statement
        else:
            final_statements[i] = clean_statement

    # --- 5. 批量生成图片嵌入 ---
    valid_paths_to_encode = []
    logging.info(f"正在安全地验证图片文件...")
    for task in tqdm(image_processing_tasks, desc="Validating images"):
        path = task['image_path']
        if path and os.path.exists(path):
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_paths_to_encode.append(path)
            except (UnidentifiedImageError, OSError):
                logging.warning(f"损坏或无法识别的图片文件，已跳过: {path}")
    
    embeddings = []
    pool = None

    clip_model_module = load_embedding_model(model_name,model_cache_dir)

    # 2. 将这个准备好的模块列表传递给 SentenceTransformer 主类进行初始化
    main_model = SentenceTransformer(
        modules=[clip_model_module],
        device='cpu',
        cache_folder=model_cache_dir
    )
    try:
        if torch.cuda.is_available() and valid_paths_to_encode:
            gpu_count = torch.cuda.device_count()
            logging.info(f"检测到 {gpu_count} 个可用的 GPU (CUDA / MUXI 兼容环境)，正在启动多GPU进程池...")
            pool = main_model.start_multi_process_pool()
            logging.info(f"多GPU进程池已启动，将处理 {len(valid_paths_to_encode)} 张有效图片。")
            embeddings = main_model.encode(valid_paths_to_encode, pool=pool, batch_size=64, show_progress_bar=True)
            
        elif valid_paths_to_encode:
            logging.info(f"无可用GPU，将在CPU上处理 {len(valid_paths_to_encode)} 张有效图片...")
            embeddings = main_model.encode(valid_paths_to_encode, batch_size=64, show_progress_bar=True)
    finally:
        if pool:
            main_model.stop_multi_process_pool(pool)
            logging.info("多GPU进程池已停止。")

    # --- 6. 结果映射与最终SQL构建 ---
    embedding_map = {path: emb for path, emb in zip(valid_paths_to_encode, embeddings)}
    for task in image_processing_tasks:
        embedding = embedding_map.get(task['image_path'], np.zeros(embedding_dim))
        task['embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

    logging.info("正在内存中构建最终的SQL命令...")
    for task in tqdm(image_processing_tasks, desc="Building final INSERTs"):
        original_index = task['original_index']
        original_statement = sql_statements[original_index].rstrip(';')
        
        full_row_data = _parse_sql_insert_line(original_statement, table_to_enhance)
        if full_row_data:
            full_row_data['image_embedding'] = str(task['embedding'])
            new_columns = full_row_data.keys()
            new_values = [_format_sql_value(v) for v in full_row_data.values()]
            new_insert_sql = f'INSERT INTO "{table_to_enhance}" ({", ".join(f"`{c}`" for c in new_columns)}) VALUES ({", ".join(new_values)})'
            final_statements[original_index] = new_insert_sql

    for i, statement in enumerate(final_statements):
        if statement and create_table_pattern.match(statement):
            col_defs_match = re.search(r'\((.*)\)', statement, re.DOTALL)
            if col_defs_match:
                col_defs = [c.strip() for c in col_defs_match.group(1).split(',')]
                clean_col_defs = [c for c in col_defs if 'image_embedding' not in c and c]
                clean_col_defs.append(f"image_embedding float[{embedding_dim}]")
                final_col_defs_str = ",\n  ".join(clean_col_defs)
                new_create_sql = f"CREATE VIRTUAL TABLE \"{table_to_enhance}\" USING vec0(\n  {final_col_defs_str}\n)"
                final_statements[i] = new_create_sql

    build_conn = None
    try:
        # --- 7. 一次性构建最终数据库 ---
        logging.info(f"正在构建最终数据库: '{output_db_path}'...")
        if os.path.exists(output_db_path):
            os.remove(output_db_path)
        
        build_conn = sqlite3.connect(output_db_path)
        build_conn.enable_load_extension(True)
        sqlite_vec.load(build_conn)
        build_cursor = build_conn.cursor()
        
        final_sql_script = ";\n".join(filter(None, final_statements)) + ";"
        build_cursor.executescript(final_sql_script)
        
        build_conn.commit()
        logging.info("✅ 最终数据库构建成功！")

    except Exception as e:
        logging.error(f"操作失败: {e}", exc_info=True)
        if build_conn: build_conn.rollback()
    finally:
        if build_conn:
            build_conn.close()

# --- 主执行块：调用示例 ---
if __name__ == '__main__':
    pass
