from typing import TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, START, END
import os
from tavily import TavilyClient

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(url="http://localhost:6333")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# The shared "clipboard" that flows through the model.
class State(TypedDict):
    question: str   # the current question; the rewrite node may change this
    chunks: list    
    answer: str     
    retries: int    # how many times we've rewritten and retried, to cap the loop
    relevant: bool   # did the grade node judge the chunks relevant?
    web_examples: str



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

def grade(state: State) -> dict:
    question = state["question"]
    chunks = state["chunks"]

    context = "\n\n".join(c["text"] for c in chunks)

    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved text:\n{context}\n\n"
        "Does the retrieved text contain information that answers the question? "
        "Reply with only one word: yes or no."
    )

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":'user', "content":prompt}
        ]
    )

    verdict = completion.choices[0].message.content.strip().lower()
    relevant = verdict.startswith("y")

    return {"relevant": relevant}


def rewrite(state: State) -> dict:
    question = state["question"]

    prompt = (
        f"A search of Go documentation for this question returned nothing useful: '{question}'. "
        "Rewrite it as a clearer search query more likely to match Go documentation. "
        "Reply with only the rewritten question."
    )

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"user", "content":prompt}
        ]
    )

    new_question = completion.choices[0].message.content.strip()

    # save new question AND bump the retry counter.
    return {"question": new_question, "retries": state["retries"] + 1}


def decide(state: State) -> str:
    if state["relevant"]:
        return "answer"          # chunks are good, go answer
    if state["retries"] >= 2:
        return "answer"          # tried enough, give up looping and answer anyway
    return "rewrite"             # chunks weak and we still have tries left, rephrase


def web_examples(state: State) -> dict:
    question = state["question"]

    # Ask Tavily for a few web results about this Go question.
    response = tavily.search(query=f"Go programming: {question}", max_results=3)

    lines = []
    for item in response["results"]:
        snippet = item["content"][:150].strip()
        lines.append(f"- [{item['title']}]({item['url']})\n  {snippet}...")

    web = "\n\n".join(lines) if lines else "No web examples found."
    return {"web_examples": web}


builder = StateGraph(State)

# Register the nodes function
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_node("grade", grade)
builder.add_node("rewrite", rewrite)
builder.add_node("web_examples", web_examples)




builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")  
builder.add_conditional_edges("grade", decide, {"answer": "answer", "rewrite": "rewrite"})  
builder.add_edge("rewrite", "retrieve")  

builder.add_edge("answer", "web_examples")  
builder.add_edge("web_examples", END)   

graph = builder.compile()

def ask_full(question):
    return graph.invoke({"question": question, "chunks": [], "answer": "", "retries": 0, "web_examples": ""})

def ask(question):
    return ask_full(question)["answer"]

if __name__ == "__main__":
    result = ask_full("How do goroutines work?")
    print(result["answer"])
    print("\n--- Examples from the web ---")
    print(result["web_examples"])