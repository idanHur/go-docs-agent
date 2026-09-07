from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(url="http://localhost:6333")
data_folder = Path("data")

chunks = []

# Loop over every .txt file in the data folder.
for file_path in data_folder.glob("*.txt"):

    text = file_path.read_text(encoding="utf-8")

    paragraphs = text.split("\n\n")

    # For each paragraph, skip empty ones and store the rest with their source file name.
    for para in paragraphs:
        if len(para.strip()) < 30:
            continue
        chunks.append({"text": para.strip(), "source": file_path.name})

print(f"Total chunks: {len(chunks)}")
for c in chunks:
    print(f"[{c['source']}] {c['text'][:80]}...")


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