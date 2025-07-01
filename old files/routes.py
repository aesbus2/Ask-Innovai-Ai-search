# routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from indexing import process_evaluation_document
from embedder import get_embedding_service
from opensearch_client import client as opensearch_client
import requests
import asyncio

router = APIRouter()

### ----------------------
### MANUAL IMPORT ENDPOINT
### ----------------------
@router.post("/import_evaluations")
async def run_eval_import():
    try:
        response = requests.get("https://your.api/evaluations")
        response.raise_for_status()
        data = response.json()

        documents = data.get("evaluations", [])
        total_chunks = 0

        for doc in documents:
            total_chunks += await process_evaluation_document(doc)

        return {
            "status": "success",
            "message": f"Imported {len(documents)} evaluations with {total_chunks} chunks."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


### ----------------------
### INSIGHT QUESTION ENDPOINT
### ----------------------
class InsightRequest(BaseModel):
    question: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    template_name: Optional[str] = None
    lob: Optional[str] = None
    site: Optional[str] = None
    agentName: Optional[str] = None
    max_docs: Optional[int] = 25

@router.post("/insight")
async def get_insight(request: InsightRequest):
    try:
        filters = []

        if request.start_date and request.end_date:
            filters.append({
                "range": {
                    "call_date": {
                        "gte": request.start_date,
                        "lte": request.end_date
                    }
                }
            })

        for field in ["template_name", "lob", "site", "agentName"]:
            value = getattr(request, field)
            if value:
                filters.append({"term": {f"{field}.keyword": value}})

        query_body = {
            "query": {"bool": {"filter": filters}},
            "size": request.max_docs,
            "_source": ["evaluation", "transcript", "agentName", "template_name", "call_date"]
        }

        response = opensearch_client.search(index="eval-*", body=query_body)
        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            return {"status": "empty", "message": "No matching evaluations found."}

        context = [
            f"Agent: {h['_source'].get('agentName')}, Date: {h['_source'].get('call_date')},\nEval: {h['_source'].get('evaluation')},\nTranscript: {h['_source'].get('transcript')}"
            for h in hits
        ]

        joined_context = "\n---\n".join(context[:request.max_docs])

        model = get_embedding_service()  # LLM interface
        response_text = model.ask_with_context(request.question, joined_context)

        return {
            "status": "success",
            "question": request.question,
            "matches": len(hits),
            "response": response_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
