import os
from PIL import Image
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

DATASET_ROOT = "human_faces_dataset"
SUBFOLDERS = ["men", "women"]
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
idx = 0
print(f"Indexing images from {DATASET_ROOT}...")

for gender_folder in SUBFOLDERS:
    folder_path = os.path.join(DATASET_ROOT, gender_folder)
    
    # Determine gender string for payload
    gender_label = "Male" if gender_folder == "men" else "Female"

    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
            image_path = os.path.join(folder_path, filename)
            
            try:
                # Convert image to vector
                image = Image.open(image_path)
                vector = model.encode(image).tolist()
                #
                #vector = model.encode(image).tolist()
                
                # Create Point 
                point = PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "filename": filename,
                        "gender": gender_label,
                        "path": image_path
                    }
                )
                points.append(point)
                idx += 1
                print(f"Processed ({gender_label}): {filename}")
                
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