from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

print(client.get_collections())