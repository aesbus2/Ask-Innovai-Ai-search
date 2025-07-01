# insight_endpoint.py

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
from opensearch_client import client as opensearch_client
from embedder import get_embedding_service

router = APIRouter()

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
            "query": {
                "bool": {
                    "filter": filters
                }
            },
            "size": request.max_docs,
            "_source": ["evaluation", "transcript", "agentName", "template_name", "call_date"]
        }

        response = opensearch_client.search(index="eval-*", body=query_body)
        hits = response.get("hits", {}).get("hits", [])

        if not hits:
            return {"status": "empty", "message": "No matching evaluations found."}

        context = []
        for h in hits:
            s = h["_source"]
            context.append(f"Agent: {s.get('agentName')}, Date: {s.get('call_date')}, Eval: {s.get('evaluation')}, Transcript: {s.get('transcript')}")

        joined_context = "\n---\n".join(context[:request.max_docs])

        model = get_embedding_service()  # Replace with LLM client
        response_text = model.ask_with_context(request.question, joined_context)

        return {
            "status": "success",
            "question": request.question,
            "matches": len(hits),
            "response": response_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
