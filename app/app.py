import os
import io
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

load_dotenv()
app = FastAPI(title="Visual Search API")

app.mount("/static", StaticFiles(directory="human_faces_dataset"), name="static")

# Global variables for Model and DB
# We load these at the top level so they are ready when the app starts
try:
    print("Loading model and connecting to Cloud DB...")
    model = SentenceTransformer('clip-ViT-B-32')
    
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    COLLECTION_NAME = "image_posts"
    print("Startup Complete.")
except Exception as e:
    print(f"Error during startup: {e}")

@app.get("/")
def home():
    return {"message": "Visual Search API is live on Qdrant Cloud!"}

@app.post("/search")
async def search_products(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # A. Read Image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # B. Encode (Image -> Vector)
        query_vector = model.encode(image).tolist()
        
        # C. Search Cloud DB
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        
        print("Result", search_result)
        # D. Format Response
        results = []            
        for hit in search_result.points:
            original_path = hit.payload.get("path")
        
            # Remove the root folder name from the path to get "men/98.jpg"
            # (Assuming your path always starts with "human_faces_dataset/")
            relative_path = original_path.replace("human_faces_dataset/", "")
            
            # Create full URL
            image_url = f"http://127.0.0.1:8001/static/{relative_path}"
            
            results.append({
                "filename": hit.payload.get("filename"),
                # FIX 2: Use 'gender' instead of 'product_name' to match your data
                "gender": hit.payload.get("gender"),
                "similarity_score": round(hit.score, 4),
                "image_url": image_url
            })
            
        return {"matches": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))