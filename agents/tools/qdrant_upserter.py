import uuid
from bs4 import BeautifulSoup
from cleantext import clean
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# ====================================
# 1. Initialize Cloud Qdrant Client
# ====================================

CLOUD_QDRANT_URL = "https://182dab79-601a-482c-9e3e-89bd4653f656.us-east-1-0.aws.cloud.qdrant.io:6333"
CLOUD_QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7KAOYyLWeLgU3rIJv5BrdmcKM7_xjjfd9tqVWJwqzas"
QDRANT_COLLECTION = "web_content"

qdrant_client = QdrantClient(
    url=CLOUD_QDRANT_URL,
    api_key=CLOUD_QDRANT_API_KEY,
)

# Initialize Google Generative AI Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key= "AIzaSyAYT5dbF_-QjFG-mSoFbIUQQ1An7nMHyE4"  # Ensure your API key is set in the environment
)

# ====================================
# 2. Ensure collection exists
# ====================================

def ensure_collection():
    existing = qdrant_client.get_collections().collections
    if QDRANT_COLLECTION in [col.name for col in existing]:
        print(f"[INFO] Deleting existing collection '{QDRANT_COLLECTION}'...")
        qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)

    print(f"[INFO] Creating collection '{QDRANT_COLLECTION}' with vector size 768...")
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=768, distance='Cosine')  # Correct dimension
    )

# ====================================
# 3. Parse and Chunk HTML
# ====================================

def extract_text_from_html(html_path: str) -> str:
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def chunk_text(text: str, chunk_size: int = 1000):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ====================================
# 4. Upload to Qdrant
# ====================================

def upload_html_to_cloud_qdrant(html_path: str, title_prefix: str = "Legal HTML Document"):
    ensure_collection()

    # Extract and clean
    raw_text = extract_text_from_html(html_path)
    cleaned_text = clean(
        raw_text,
        fix_unicode=True,
        to_ascii=True,
        lower=True,
        no_line_breaks=True,
        lang="en"
    )

    # Chunk the text
    chunks = chunk_text(cleaned_text)

    # Embed and upsert
    for idx, chunk in enumerate(chunks):
        embedding = embedding_model.embed_query(text=chunk)
        point_id = str(uuid.uuid4())

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[{
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "title": f"{title_prefix} - Part {idx+1}",
                    "content": chunk,
                    "url": "local_html_doc",
                    "source": "html_legal"
                }
            }]
        )

    print(f"[INFO] Successfully uploaded {len(chunks)} chunks from {html_path} to Cloud Qdrant!")

# ====================================
# 5. Main Execution
# ====================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "National Survey of State Laws - HeinOnline.org.html")
    upload_html_to_cloud_qdrant(html_path)