from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(url="http://localhost:6333")
data_folder = Path("data")

CHUNK_STRATEGY = "paragraph" #"bysize"    
CHUNK_SIZE = 900    

chunks = []

# Fresh slate every run, so old chunks from a previous size do not linger.
if qdrant.collection_exists("go_docs"):
    qdrant.delete_collection("go_docs")
qdrant.create_collection("go_docs", vectors_config=VectorParams(size=1536, distance=Distance.COSINE))


def chunk_by_paragraph(text):
    chunks = []
    for para in text.split("\n\n"):
        if len(para.strip()) >= 30:
            chunks.append(para.strip())
    return chunks

def chunk_by_size(text, size, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + size]
        if len(piece.strip()) >= 30:
            chunks.append(piece.strip())
        start += size - overlap
    return chunks

def make_chunks(text):
    if CHUNK_STRATEGY == "paragraph":
        return chunk_by_paragraph(text)
    else:
        return chunk_by_size(text, CHUNK_SIZE)
    

# Loop over every .txt file in the data folder.
for file_path in data_folder.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")
    for piece in make_chunks(text):
        chunks.append({"text": piece, "source": file_path.name})

print(f"Total chunks: {len(chunks)}")


points = []
for i, chunk in enumerate(chunks):
    # embed this chunk's text. 
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=chunk["text"])
    vector = response.data[0].embedding

    # Build one Qdrant point
    point = PointStruct(id=i, vector=vector, payload=chunk)

    points.append(point)


# Store every point in the collection in a single call
qdrant.upsert(collection_name="go_docs", points=points)


print(f"Stored {len(points)} points in Qdrant.")
