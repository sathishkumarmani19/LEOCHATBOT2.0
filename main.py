import os
import sys
import logging
from groq import Groq
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chromadb.utils import embedding_functions

# --- RENDER SQLITE FIX (Required for deployment) ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass 

app = FastAPI()

# GREETING CONSTANT
EXACT_GREETING = (
    "Hello! I am Leo Bot, your HITS Expert. I'm delighted to provide you with "
    "detailed and professional information regarding Hindustan Institute of Technology "
    "and Science (HITS), particularly focusing on HITSEEE, Admissions, and the "
    "esteemed Department of Aeronautical Engineering."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GROQ INITIALIZATION ---
# Ensure your GROQ_API_KEY is set in your environment variables
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- VECTOR DB INITIALIZATION ---
# We use try/except to handle connection errors gracefully
try:
    # 1. Path must match your renamed folder
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    # 2. Collection name MUST match your ingest.py (verified as 'hits_web_data' in audit)
    # Using get_collection will error if it doesn't exist, helping you debug.
    collection = db_client.get_collection(name="hits_web_data", embedding_function=default_ef)
    print("✅ Connected to hits_web_data collection successfully.")
except Exception as e:
    print(f"❌ DB Connection Error: {e}")
    # Fallback to prevent app crash on startup if folder is missing during first deploy
    collection = None

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        if not collection:
            return {"response": "System initializing. Please try again in a few seconds."}

        clean_query = query.text.lower().strip()
        
        # 1. Handling Greetings
        if clean_query in ["hi", "hello", "hey", "start", "greetings"]:
            return {"response": EXACT_GREETING}

        # 2. Retrieve Relevant Context
        # We fetch 5 chunks to ensure we get detailed lab/date info
        results = collection.query(
            query_texts=[clean_query], 
            n_results=5, 
            include=['documents']
        )
        
        context = "\n".join(results['documents'][0]) if results['documents'] else ""

        # 3. Generate Response with Groq (Llama 3.3 70B)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are Leo Bot, the HITS Expert. {EXACT_GREETING} Always use the following context to answer precisely. If the information is not in the context, kindly state you don't have that specific info and suggest contacting info@hindustanuniv.ac.in."
                },
                {
                    "role": "user",
                    "content": f"CONTEXT: {context}\n\nQUESTION: {query.text}"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2, # Low temperature for factual accuracy
        )
        
        return {"response": chat_completion.choices[0].message.content}

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return {"response": "I'm currently having trouble accessing my knowledge base. Please contact admission office at 1800 425 44 38."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
