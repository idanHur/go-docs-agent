from dotenv import load_dotenv
from agent import ask, openai_client, qdrant

load_dotenv()

test_set = [
    {"question": "How do I start a goroutine?", "expected_source": "goroutines.txt"},
    {"question": "What is a channel used for?", "expected_source": "goroutines.txt"},
    {"question": "How does append grow a slice?", "expected_source": "slices.txt"},
    {"question": "What are the length and capacity of a slice?", "expected_source": "slices.txt"},
    {"question": "How do I wrap an error with context?", "expected_source": "errors.txt"},
    {"question": "What does errors.Is do?", "expected_source": "errors.txt"},
    {"question": "Does goroutines communicate between one another?", "expected_source": "goroutines.txt"},
    # Phrased without the obvious keyword (tests real semantic matching, not word overlap):
    {"question": "Why does my program exit before my background work finishes?", "expected_source": "goroutines.txt"},
    {"question": "How do I make an independent duplicate of a list so edits don't leak?", "expected_source": "slices.txt"},
    {"question": "What is the idiomatic way for a function to signal something went wrong?", "expected_source": "errors.txt"},
    # Uses a word that appears in more than one file (tests discrimination):
    {"question": "How does the Go runtime manage many concurrent tasks cheaply?", "expected_source": "goroutines.txt"},
    # Borderline / could genuinely miss (that's fine, it's honest):
    {"question": "How do I check what kind of failure I got back?", "expected_source": "errors.txt"},
        # maps
    {"question": "How do I check whether a key is present without a false zero?", "expected_source": "maps.txt"},
    {"question": "Why do my key-value pairs come out in a different order each run?", "expected_source": "maps.txt"},
    # interfaces
    {"question": "How can unrelated types be used interchangeably when they share the same methods?", "expected_source": "interfaces.txt"},
    {"question": "How does a type conform to a contract without declaring that it does?", "expected_source": "interfaces.txt"},
    # structs
    {"question": "How do I define a custom type that groups several named fields?", "expected_source": "structs.txt"},
    # pointers
    {"question": "How do I let a function change the caller's variable instead of a copy?", "expected_source": "pointers.txt"},
    {"question": "What holds the memory address of another value?", "expected_source": "pointers.txt"},
    # defer
    {"question": "How do I make sure a file closes no matter how the function exits?", "expected_source": "defer.txt"},
    {"question": "In what order do scheduled cleanup calls run?", "expected_source": "defer.txt"},
    # "methods" is taught in BOTH structs.txt and pointers.txt, so these are genuinely ambiguous:
    {"question": "How do I attach behavior to my own data type?", "expected_source": "structs.txt"},
    {"question": "Should a method take a value or a pointer receiver?", "expected_source": "pointers.txt"},
    # embedding vs interfaces overlap:
    {"question": "How can one type reuse another type's fields and methods without inheritance?", "expected_source": "structs.txt"},
]

def judge(question, answer):
    prompt = (
        "You are grading an answer about the Go programming language.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Rate the answer from 1 to 5, where 5 means fully correct and complete, "
        "and 1 means wrong or irrelevant. Reply with only the single digit."
    )

    completion = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    text = completion.choices[0].message.content.strip()
    digits = [c for c in text if c in "12345"]
    return int(digits[0]) if digits else 0


def search_sources(question, limit=3):
    vector = openai_client.embeddings.create(
        model="text-embedding-3-small", input=question
    ).data[0].embedding
    results = qdrant.query_points(
        collection_name="go_docs", query=vector, limit=limit
    ).points
    return [r.payload["source"] for r in results]


hits_at_1 = 0
hits_at_3 = 0
for case in test_set:
    sources = search_sources(case["question"], limit=3)
    expected = case["expected_source"]

    hit_at_1 = sources[0] == expected
    hit_at_3 = expected in sources

    if hit_at_1:
        hits_at_1 += 1
    if hit_at_3:
        hits_at_3 += 1
        
    mark = "PASS" if hit_at_1 else ("top3" if hit_at_3 else "FAIL")
    print(f"{mark:5} {case['question']}  ->  {sources}")



n = len(test_set)
print(f"\nRecall@1: {hits_at_1}/{n} = {hits_at_1 / n:.0%}")
print(f"Recall@3: {hits_at_3}/{n} = {hits_at_3 / n:.0%}")


# --- Answer-quality metric (runs the full agent per question) ---

total = 0
for case in test_set:
    a = ask(case["question"])          
    total += judge(case["question"], a)


print(f"\nAverage answer quality: {total / n:.2f} / 5")