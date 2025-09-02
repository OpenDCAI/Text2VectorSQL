import os
import json
import requests
import time
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image # 导入Pillow库的Image模块

# ==============================================================================
# --- 配置区 (CONFIGURATION) ---
# ==============================================================================

# 1. 数据集路径
DATASET_BASE_PATH = '/mnt/b_public/data/wangzr/Text2VectorSQL/database/wiki'

# 2. 网络请求与下载参数
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
MAX_RETRIES = 5
RETRY_BACKOFF_FACTOR = 1
STATUS_FORCELIST = [429, 500, 502, 503, 504]
PAGE_TIMEOUT = 20
IMAGE_TIMEOUT = 60
DOWNLOAD_CHUNK_SIZE = 8192

# 3. 并发控制
MAX_WORKERS = 4

import cairosvg
from PIL import Image
def convert_svg_to_jpg_alternative(svg_path, jpg_path):
    png_path = jpg_path.with_suffix('.png')
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    with Image.open(png_path) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(jpg_path, 'JPEG')
    os.remove(png_path)

# ==============================================================================
# --- 主逻辑 (MAIN LOGIC) ---
# ==============================================================================

def create_session_with_retries():
    """
    创建一个带有自动重试机制的 requests.Session 对象。
    """
    s = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=STATUS_FORCELIST,
        allowed_methods={"HEAD", "GET", "OPTIONS"}
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    s.headers.update({'User-Agent': USER_AGENT})
    return s

def scrape_and_download_from_meta(session, image_info, output_dir):
    """
    从元信息中获取维基百科文件页面URL，抓取真实图片链接并下载。
    (新增功能：对已存在的文件进行完整性校验)
    """
    if 'url' not in image_info or 'filename' not in image_info:
        return f"元信息格式错误: {image_info}"

    page_url = image_info['url']
    image_filename = image_info['filename']
    output_path = output_dir / image_filename

    # --- 关键修改：检查已存在文件的有效性 ---
    # 1. 如果文件已存在，则尝试打开并校验它
    if output_path.exists():
        try:
            # 使用 'with' 语句确保图片文件被正确关闭
            with Image.open(output_path) as img:
                img.verify()  # 校验图片文件结构，检查是否有损坏
            # 如果代码能执行到这里，说明图片是有效的
            return f"文件已存在且有效: {image_filename}"
        except Exception as e:
            # 捕获所有可能的异常 (如 UnidentifiedImageError, DecompressionBombError等)
            # 这表明文件已损坏或不是一个有效的图片文件，需要重新下载
            print(f"\n警告: 文件 {image_filename} 存在但已损坏或无法打开 (错误: {e})。将尝试重新下载。")
            # 不返回，继续执行后续的下载逻辑

    direct_image_url = ''
    try:
        # 2. 访问文件页面 (File:...)
        page_response = session.get(page_url, timeout=PAGE_TIMEOUT)
        page_response.raise_for_status()

        # 3. 解析HTML，找到原始图片的直接链接
        soup = BeautifulSoup(page_response.text, 'html.parser')
        link_element = soup.select_one('#file a')

        if not link_element or not link_element.has_attr('href'):
            return f"抓取失败: 在页面 {page_url} 上找不到图片链接"

        # 4. 提取并构建完整的图片URL
        direct_image_url = link_element['href']
        if direct_image_url.startswith('//'):
            direct_image_url = 'https:' + direct_image_url

        # 5. 下载图片
        image_response = session.get(direct_image_url, stream=False, timeout=IMAGE_TIMEOUT)
        image_response.raise_for_status()

        # 6. 保存到文件
        with open(output_path, 'wb') as f:
            for chunk in image_response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
        
        # 7. 转换SVG格式图片为JPG格式
        if direct_image_url.endswith('.svg'):
            convert_svg_to_jpg_alternative(output_path.with_suffix('.jpg'), output_path.with_suffix('.jpg'))
            print(f"已转换SVG为JPG: {image_filename} -> {output_path.with_suffix('.jpg').name}")
            pass

        return f"下载成功: {image_filename}"

    except requests.exceptions.RequestException as e:
        return f"请求最终失败: {e} (URL: {direct_image_url or page_url})"
    except Exception as e:
        return f"下载或保存时出错: {e} (文件: {image_filename})"


def process_article_directory(article_dir):
    """
    处理单个文章目录，读取 meta.json 并下载所有图片。
    """
    meta_file = article_dir / 'img/meta.json'
    if not meta_file.is_file():
        return

    img_output_dir = article_dir / 'img'
    img_output_dir.mkdir(exist_ok=True)

    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            images_metadata_str = json.load(f)
            images_metadata_list = json.loads(images_metadata_str).get('img_meta', [])
    except (json.JSONDecodeError, TypeError, FileNotFoundError) as e:
        print(f"错误: 无法解析或格式不正确 '{meta_file}' ({e})。")
        return
            
    if not images_metadata_list:
        return

    session = create_session_with_retries()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        progress_bar = tqdm(total=len(images_metadata_list), desc=f"处理中 {article_dir.name}", leave=False)
        
        # future_to_info = {
        #     executor.submit(scrape_and_download_from_meta, session, img_info, img_output_dir): img_info 
        #     for img_info in images_metadata_list
        # }
        
        # for future in as_completed(future_to_info):
        #     # result = future.result() # 可以取消注释来调试
        #     progress_bar.update(1)

        for img_info in images_metadata_list:
            scrape_and_download_from_meta(session, img_info, img_output_dir)
            progress_bar.update(1)

        progress_bar.close()


def main():
    """
    主函数，启动数据集图片下载流程。
    """
    base_dir = Path(DATASET_BASE_PATH)

    if not base_dir.is_dir():
        print(f"错误: 目录 '{DATASET_BASE_PATH}' 不存在或不是一个有效的目录。")
        return

    article_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not article_dirs:
        print(f"错误: 在 '{base_dir}' 下未找到任何文章子目录。请检查路径是否正确。")
        return

    print(f"在数据集中共找到 {len(article_dirs)} 个文章目录。")
    print("开始下载图片（将校验已存在的图片）...")

    for article_dir in tqdm(article_dirs, desc="总进度"):
        process_article_directory(article_dir)

    print("\n所有图片下载任务完成！")


if __name__ == '__main__':
    main()