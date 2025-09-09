import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Dict

import openai
from dotenv import load_dotenv  # <-- Import dotenv
from tqdm import tqdm

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

def synthesize_questions(
    input_file: str,
    output_file: str,
    model_name: str,
    api_key: str,
    api_url: str,
    max_workers: int
):
    """
    Runs the main logic for question synthesis from a dataset.

    Args:
        input_file (str): Path to the input JSON file containing prompts.
        output_file (str): Path to save the output JSON file with responses.
        model_name (str): Name of the LLM model to use for inference.
        api_key (str): The API key for the LLM service.
        api_url (str): The base URL for the LLM API endpoint.
        max_workers (int): The number of parallel threads for inference.
    """
    # Validate that essential variables are provided
    if not api_key or not model_name:
        raise ValueError("Error: api_key and model_name must be provided.")
    
    print("--- Running Synthesis with Configuration ---")
    print(f"Model: {model_name}")
    print(f"API URL: {api_url}")
    print(f"Max Workers: {max_workers}")
    print(f"Input File: {input_file}")
    print(f"Output File: {output_file}")
    print("------------------------------------------")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_dataset = json.load(f)
    
    results = llm_inference(
        model=model_name,
        dataset=input_dataset,
        api_key=api_key,
        api_url=api_url,
        parallel_workers=max_workers
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSynthesis complete. Results saved to {output_file}")


if __name__ == '__main__':
    # This block now serves as the script's entry point.
    # It parses command-line arguments and loads environment variables,
    # then calls the main logic function with the collected configuration.
    parser = argparse.ArgumentParser(description="Run LLM inference for question synthesis.")
    parser.add_argument("--input_file", type=str, default="./prompts/question_synthesis_prompts.json")
    parser.add_argument("--output_file", type=str, default="./results/question_synthesis.json")
    
    opt = parser.parse_args()

    # Load configuration from environment variables
    # This keeps the convenience of using a .env file when running the script directly.
    api_key_env = os.getenv("API_KEY")
    api_url_env = os.getenv("BASE_URL")
    model_name_env = os.getenv("LLM_MODEL_NAME")
    max_workers_env = int(os.getenv("MAX_WORKERS", 4))

    # Call the main function with the loaded configuration
    synthesize_questions(
        input_file=opt.input_file,
        output_file=opt.output_file,
        model_name=model_name_env,
        api_key=api_key_env,
        api_url=api_url_env,
        max_workers=max_workers_env
    )
