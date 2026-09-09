from typing import TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(url="http://localhost:6333")

# The shared "clipboard" that flows through the model.
class State(TypedDict):
    question: str   # the current question; the rewrite node may change this
    chunks: list    
    answer: str     
    retries: int    # how many times we've rewritten and retried, to cap the loop



def retrieve(state: State) -> dict:
    question = state["question"]

    vector = openai_client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    results = qdrant.query_points(collection_name="go_docs", query=vector, limit=3).points

    chunks = []
    for r in results:
        chunks.append({"text": r.payload["text"], "source": r.payload["source"]})

    # Returns ONLY what changed in the state.
    return {"chunks": chunks}


def answer(state: State) -> dict:
    question = state["question"]
    chunks = state["chunks"]

    context = ""

    for i, c in enumerate(chunks):
        context += f"[Source {i + 1}: {c['source']}]\n{c['text']}\n\n"

    system_prompt = (
        "You answer questions about the Go programming language. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Always cite the source file(s) you used."
    )

    user_prompt = f"Context:\n{context}\nQuestion: {question}"

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
          {"role": "system", "content": system_prompt},
            {"role": "user", "content":user_prompt},
        ],
    )

    return {"answer": completion.choices[0].message.content}



test_state = {"question": "How do goroutines work?", "chunks": [], "answer": "", "retries": 0}
print(retrieve(test_state))