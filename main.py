import os
import sys
import logging
import time

# --- 1. RENDER SQLITE VERSION FIX ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass 

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai 
from chromadb.utils import embedding_functions

# Optimization for Render CPU
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

# --- 2. LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HITS_LEO_BOT")

app = FastAPI()

# --- 3. CONSTANTS & PERSONA ---
EXACT_GREETING = (
    "Hello! I am Leo Bot, your HITS Expert. I'm delighted to provide you with "
    "detailed and professional information regarding Hindustan Institute of Technology "
    "and Science (HITS), particularly focusing on HITSEEE, Admissions, and the "
    "esteemed Department of Aeronautical Engineering."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. DUAL API KEY INITIALIZATION ---
# Add GOOGLE_API_KEY_PRIMARY and GOOGLE_API_KEY_SECONDARY to Render Env Vars
API_KEYS = [
    os.getenv("GOOGLE_API_KEY_PRIMARY"),
    os.getenv("GOOGLE_API_KEY_SECONDARY")
]

# Create multiple clients for failover using v1beta for best compatibility
clients = []
for key in API_KEYS:
    if key:
        clients.append(genai.Client(api_key=key, http_options={'api_version': 'v1beta'}))

# Initialize Vector DB
try:
    db_client = chromadb.PersistentClient(path="./hits_vectordb")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = db_client.get_collection(name="hits_knowledge", embedding_function=default_ef)
    logger.info("Vector Database loaded successfully.")
except Exception as e:
    logger.error(f"DB Error: {e}")

class Query(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"status": "online", "bot": "HITS Leo Bot 2.0 Combined"}

# --- 5. MAIN CHAT ENDPOINT ---
@app.post("/chat")
async def chat(query: Query):
    try:
        clean_query = query.text.lower().strip()
        
        # A. Handled Exact Greeting
        if clean_query in ["hi", "hello", "hey", "start", "greetings"]:
            return {"response": EXACT_GREETING}

        # B. Search Vector DB
        results = collection.query(
            query_texts=[clean_query], 
            n_results=5, 
            include=['documents', 'distances']
        )
        
        best_distance = results['distances'][0][0] if results['distances'] else 2.0
        
        # C. Context Construction (Distance 1.7 is good for precision)
        if best_distance < 1.7:
            context = "\n".join(results['documents'][0])
            persona_prefix = (
                f"You are Leo Bot, the HITS Expert. {EXACT_GREETING}\n\n"
                "INSTRUCTIONS: Use the context below to answer. Use markdown tables for data. "
                "If not in context, refer to info@hindustanuniv.ac.in.\n\n"
                f"Context: {context}"
            )
            full_prompt = f"{persona_prefix}\n\nUser Question: {query.text}"
        else:
            return {"response": "I'm sorry, I don't have that specific information in my records. Please contact **info@hindustanuniv.ac.in** for assistance."}

        # D. Dual-Key & Multi-Model Failover Loop
        # Using fully qualified names to avoid 404s
        model_priority = ["models/gemini-1.5-flash", "models/gemini-1.5-flash-8b"]

        for client_idx, gen_client in enumerate(clients):
            for model_id in model_priority:
                try:
                    logger.info(f"Attempting Client {client_idx + 1} with {model_id}")
                    response = gen_client.models.generate_content(
                        model=model_id,
                        contents=full_prompt
                    )
                    if response:
                        return {"response": response.text}
                
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Fail: Client {client_idx + 1}, Model {model_id}: {error_msg}")
                    
                    # If hit a Rate Limit (429), pause briefly before trying next
                    if "429" in error_msg:
                        time.sleep(1)
                    continue 

        return {"response": "All HITS system nodes are currently busy. Please try again in a moment or email **info@hindustanuniv.ac.in**."}

    except Exception as final_e:
        logger.critical(f"Critical System Error: {str(final_e)}")
        return {"response": "A system error occurred. Please contact the administrator at **info@hindustanuniv.ac.in**."}
