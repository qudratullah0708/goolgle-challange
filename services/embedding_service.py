from PIL import Image
import io, base64
from cohere import Client as CohereClient
from services.qdrant_service import store_embedding
from uuid import uuid4
import os
from google import genai
from dotenv import load_dotenv
load_dotenv() 


MAX_PIXELS = 1568 * 1568
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = CohereClient(api_key=COHERE_API_KEY)

def resize_image(pil_image: Image.Image) -> Image.Image:
    w, h = pil_image.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        pil_image = pil_image.resize((int(w * scale), int(h * scale)))
    return pil_image

def pil_to_base64(pil_image: Image.Image) -> str:
    pil_image = resize_image(pil_image)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    base64_img = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{base64_img}"

def compute_image_embedding(base64_img: str) -> list:
    response = co.embed(
        model="embed-v4.0",
        input_type="image",
        embedding_types=["float"],
        images=[base64_img]
    )
    return response.embeddings.float[0]

def compute_query_embedding(query: str) -> list:
    response = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        texts=[query],
        embedding_types=["float"]
    )
    return response.embeddings.float[0]
