import argparse
import json
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import openai
import os
from typing import List, Dict
from dotenv import load_dotenv # <-- Import dotenv

# Load environment variables from a .env file
load_dotenv()

@lru_cache(maxsize=10000)
def cached_llm_call(model: str, prompt: str, api_url: str, api_key: str) -> str:
    """
    Cached LLM call to avoid redundant API requests for same prompts.
    """
    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_url if api_url else None
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return ""

def llm_inference(
    model: str, 
    dataset: List[Dict], 
    api_key: str, 
    api_url: str = "", 
    parallel_workers: int = 4
) -> List[Dict]:
    """
    Perform LLM inference with caching and parallel processing.
    
    Args:
        model: LLM model name
        dataset: List of input data dictionaries
        api_key: OpenAI API key
        api_url: Custom API URL (optional)
        parallel_workers: Number of parallel workers
    
    Returns:
        List of results with generated responses
    """
    def process_item(data: Dict) -> Dict:
        prompt = data["prompt"]
        response = cached_llm_call(model, prompt, api_url, api_key)
        return {**data, "responses": [response]}  # Wrap response in list to match original format
    
    if parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            results = list(tqdm(
                executor.map(process_item, dataset),
                total=len(dataset),
                desc="Generating responses"
            ))
    else:
        results = [process_item(data) for data in tqdm(dataset, desc="Generating responses")]
    
    return results

if __name__ == '__main__':
    # --- MODIFICATION START ---
    # Arguments for model, api_key, api_url, and parallel are removed from argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./prompts/question_synthesis_prompts.json")
    parser.add_argument("--output_file", type=str, default="./results/question_synthesis.json")
    
    opt = parser.parse_args()

    # Load configuration from environment variables
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("BASE_URL")
    model_name = os.getenv("LLM_MODEL_NAME")
    # Load MAX_WORKERS, cast to int, and provide a default value
    max_workers = int(os.getenv("MAX_WORKERS", 4))

    # Validate that essential variables are loaded
    if not api_key or not model_name:
        raise ValueError("Error: API_KEY and LLM_MODEL_NAME must be set in your .env file.")
    
    print("--- Configuration Loaded from .env ---")
    print(f"Model: {model_name}")
    print(f"API URL: {api_url}")
    print(f"Max Workers: {max_workers}")
    print("--------------------------------------")
    # --- MODIFICATION END ---
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)
    
    input_dataset = json.load(open(opt.input_file, encoding="utf-8"))
    
    results = llm_inference(
        model=model_name, # Use variable from .env
        dataset=input_dataset,
        api_key=api_key, # Use variable from .env
        api_url=api_url, # Use variable from .env
        parallel_workers=max_workers # Use variable from .env
    )
    
    with open(opt.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
