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

# build one context string from the retrieved chunks.
context = ""
for i, r in enumerate(results):
    context += f"[Source {i + 1}: {r.payload['source']}]\n{r.payload['text']}\n\n"

# The instructions for the model
system_prompt = (
    "You answer questions about the Go programming language. "
    "Answer ONLY using the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Always cite the source file(s) you used."
)

# The request: the context plus the question.
user_prompt = f"Context:\n{context}\nQuestion: {question}"


completion = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":user_prompt},
    ],
)


answer =completion.choices[0].message.content

print(answer)