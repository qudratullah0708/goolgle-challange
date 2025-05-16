from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from schemas import QuestionRequest
from services.embedding_service import pil_to_base64, compute_image_embedding, compute_query_embedding
from services.qdrant_service import store_embedding, search_user_embeddings, ensure_collection_exists
from PIL import Image
import fitz
import os
from google import genai
from dotenv import load_dotenv
load_dotenv() 


genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


app = FastAPI(title="Vision RAG API")
ensure_collection_exists()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# def describe_image_with_gemini(base64_img: str) -> str:
   
#     response = genai_client.models.generate_content(
#         model="gemini-2.5-flash-preview-04-17",
#         contents=[
#             {"mime_type": "image/png", "data": base64_img},
#             {"text": "Describe this image in few sentences."}
#         ]
#     )
#     return response.text.strip()

def describe_image_with_gemini(image: Image) -> str:
        
    my_file = image
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, "Caption this image."],
    )

    print(response.text)
    return response.text.strip()




# ----- Embed Image -----
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        image = Image.open(file.file)
        base64_img = pil_to_base64(image) 
        embedding = compute_image_embedding(base64_img)
        caption = describe_image_with_gemini(image)

        # caption = describe_image_with_gemini(base64_img)
        store_embedding(user_id, embedding, {
    "source": "image",
    "filename": file.filename,
    "caption": caption
})
        return {"status": "stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----- Embed PDF -----
@app.post("/embed-pdf")
async def embed_pdf(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        for idx, page in enumerate(doc.pages()):
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            base64_img = pil_to_base64(img)
            emb = compute_image_embedding(base64_img)
            caption = describe_image_with_gemini(img)
            #caption = describe_image_with_gemini(base64_img)
            store_embedding(user_id, emb, {"source": "pdf", "page": idx, "caption":caption})
        return {"status": "stored", "pages": len(doc)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----- Ask Question -----
@app.post("/ask")
def ask_question(payload: QuestionRequest):
    try:
        query_embedding = compute_query_embedding(payload.question)
        results = search_user_embeddings(payload.user_id, query_embedding)

        context = "\n".join([f"[Image {i.id} metadata: {i.payload}]" for i in results])
        prompt = f"{context}\n\nQ: {payload.question}\nA:"

        
        
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=prompt
        )
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


