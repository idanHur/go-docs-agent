from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(url="http://localhost:6333")

question = "How does Go decide which goroutine runs?"

response = openai_client.embeddings.create(model="text-embedding-3-small", input=question)
vector = response.data[0].embedding

# search Qdrant for the chunks closest to that vector.
results =qdrant.query_points(collection_name="go_docs", query=vector, limit=3).points

for r in results:
    print(f"score={r.score:.3f}  source={r.payload['source']}")
    print(r.payload["text"])
    print("-" * 40)