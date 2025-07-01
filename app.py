# Filename: app.py (original)
# Version: 1.0.0

import os
import logging
import requests
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from uuid import uuid4

from sentence_splitter import split_into_sentences, split_into_chunks
from opensearch_client import search_opensearch, index_document

load_dotenv()

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup logging
logger = logging.getLogger("ask-innovai")
logging.basicConfig(level=logging.INFO)

# Environment configuration
GENAI_ENDPOINT = os.getenv("GENAI_ENDPOINT", "https://api.digitalocean.com/v1/ai/chat")
GENAI_ACCESS_KEY = os.getenv("GENAI_ACCESS_KEY", "ask-innovai-model")

# Pydantic model for incoming chat requests
class ChatRequest(BaseModel):
    message: str
    history: list
    programs: list = []

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/chat.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/embedder-status")
async def embedder_status():
    return {"status": "ready", "model": "all-MiniLM-L6-v2"}

@app.get("/search")
async def search(q: str):
    results = search_opensearch(q)
    return JSONResponse(content={"status": "success", "results": results})

@app.post("/chat")
async def chat_handler(request: ChatRequest):
    payload = {
        "model": GENAI_ACCESS_KEY,
        "messages": [
            {"role": m["role"], "content": m["content"]} for m in request.history
        ] + [{"role": "user", "content": request.message}]
    }
    try:
        response = requests.post(
            GENAI_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GENAI_ACCESS_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"reply": reply.strip() if reply else "Sorry, I couldn't generate a response."}
    except Exception as e:
        logger.error(f"Chat endpoint failed: {str(e)}")
        return {"reply": "Error generating response. Please try again."}

@app.post("/process_document")
async def process_document(request: Request):
    try:
        body = await request.json()
        evaluation = body.get("evaluations", [])[0]
        transcript = evaluation.get("transcript", "")
        if not transcript:
            return {"status": "error", "message": "Transcript not provided."}

        soup = BeautifulSoup(transcript, "html.parser")
        raw_text = soup.get_text(" ", strip=True)
        chunks = split_into_chunks(raw_text)

        meta = {
            "evaluation_id": evaluation.get("evaluationId"),
            "template": evaluation.get("template_name"),
            "program": evaluation.get("partner"),
            "site": evaluation.get("site"),
            "lob": evaluation.get("lob"),
            "agent": evaluation.get("agentName"),
            "disposition": evaluation.get("disposition"),
            "sub_disposition": evaluation.get("subDisposition"),
            "language": evaluation.get("language"),
            "call_date": evaluation.get("call_date"),
            "call_duration": evaluation.get("call_duration")
        }

        doc_id = str(evaluation.get("internalId", uuid4()))
        collection_name = evaluation.get("template_name", "default")

        for i, chunk in enumerate(chunks):
            doc_body = {
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk["text"],
                "offset": chunk["offset"],
                "length": chunk["length"],
                "metadata": meta,
                "source": "evaluation_json"
            }
            index_document(f"{doc_id}-{i}", doc_body, index_override=collection_name)

        return {"status": "success", "chunks_indexed": len(chunks), "document_id": doc_id, "collection": collection_name}

    except Exception as e:
        logger.error(f"Failed to process document: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "components": {
            "genai": {
                "status": "connected" if GENAI_ACCESS_KEY else "not configured",
                "endpoint": GENAI_ENDPOINT,
                "note": "Ensure GENAI_ACCESS_KEY is set in environment."
            },
            "opensearch": {
                "status": "connected" if os.getenv("OPENSEARCH_HOST") else "not configured",
                "version": os.getenv("OPENSEARCH_VERSION", "unknown")
            }
        }
    }
