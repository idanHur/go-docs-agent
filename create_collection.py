from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

# Create a collection named "go_docs".
# Call client.create_collection(...) with these two arguments:
#   collection_name="go_docs"
#   vectors_config=VectorParams(size=???, distance=Distance.COSINE)
# For size, use the number of dimensions our embedding model produces.
# You saw that exact number when you ran embed_test.py.
client.create_collection(collection_name="go_docs", vectors_config=VectorParams(size=1536, distance=Distance.COSINE))