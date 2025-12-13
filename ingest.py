import os
from PIL import Image
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

IMAGE_FOLDER = "dummy_post_images"
COLLECTION_NAME = "image_posts"

# Qdrant Setup
load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url or not qdrant_key:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing from .env")

print("Connecting to Qdrant Cloud...")
client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

# Create Collection (Reset if exists)
# Note: In production, you wouldn't delete the collection every time.
if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

# Initialize Model & Cloud Client
print("Loading CLIP model...")
model = SentenceTransformer('clip-ViT-B-32')

# Process Images
points = []
print(f"Indexing images from {IMAGE_FOLDER}...")

idx = 0
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
        path = os.path.join(IMAGE_FOLDER, filename)
        
        try:
            # Convert image to vector
            image = Image.open(path)
            vector = model.encode(image).tolist()
            
            # Create a Point
            point = PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "filename": filename,
                    "product_name": f"Item: {filename}"
                }
            )
            points.append(point)
            idx += 1
            print(f"Processed {filename}")
        except Exception as e:
            print(f"Skipping {filename}: {e}")

# Upload to Cloud
if points:
    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points
    )
    print(f"SUCCESS: Uploaded {len(points)} items to Qdrant Cloud!")
else:
    print("No images found or processed.")