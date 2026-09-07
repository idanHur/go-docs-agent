from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client =OpenAI()

response = client.embeddings.create(model="text-embedding-3-small", input="Goroutines are lightweight threads managed by the Go runtime.")

vector =response.data[0].embedding

print(f'the len is: {len(vector)}, the 5 first items: {vector[:5]}')