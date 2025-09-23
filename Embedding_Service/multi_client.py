# client_example.py
import requests
import base64
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/embed"

def get_text_embedding(text: str, model: str):
    """Gets embedding for a single text."""
    payload = {
        "model": model,
        "texts": [text]
    }
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()['embeddings'][0]

def get_image_embedding(image_path: str, model: str):
    """Gets embedding for a single image file."""
    # Read image and convert to Base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    b64_string = base64.b64encode(image_bytes).decode("utf-8")
    
    payload = {
        "model": model,
        "images": [b64_string]
    }
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()['embeddings'][0]

if __name__ == "__main__":
    # Create a dummy image for testing
    try:
        img = Image.new('RGB', (60, 30), color = 'red')
        # img.save('test_image.png')
        print("Created 'test_image.png' for testing.")

        # --- Test Text Embedding ---
        print("\n--- Testing Text Embedding ---")
        text_emb = get_text_embedding("A photo of a red square", model="sentence-transformers/clip-ViT-B-32")
        print(f"Model: clip-vit-base")
        print(f"Text: 'A photo of a red square'")
        print(f"Embedding vector (first 5 dims): {text_emb[:5]}")
        print(f"Embedding dimension: {len(text_emb)}")

        # --- Test Image Embedding ---
        print("\n--- Testing Image Embedding ---")
        image_emb = get_image_embedding("test_image.png", model="sentence-transformers/clip-ViT-B-32")
        print(f"Model: clip-vit-base")
        print(f"Image: 'test_image.png'")
        print(f"Embedding vector (first 5 dims): {image_emb[:5]}")
        print(f"Embedding dimension: {len(image_emb)}")

    except requests.exceptions.ConnectionError as e:
        print(f"\nCould not connect to the server at {API_URL}.")
        print("Please ensure the embedding_server.py is running.")
    except Exception as e:
        print(f"An error occurred: {e}")